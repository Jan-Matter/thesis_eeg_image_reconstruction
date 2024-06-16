from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import _cov
import scipy

class Denoising(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass

class MVNN(Denoising):

    def __init__(self, mvnn_dim="time", **kwargs):
        super().__init__(**kwargs)
        self.__mvnn_dim = mvnn_dim

    def transform(self, data):
        sigma_inv = self.__compute_cov_mat(data["epoched_eeg_data"])
        data["epoched_eeg_data"] = self.__whiten_data(data["epoched_eeg_data"], sigma_inv)
        return data
    

    def __compute_cov_mat(self, data):
        ### Compute the covariance matrices ###
        # covariance matrix of shape:
        # EEG channels × EEG channels
        sigma = np.empty((data[0].shape[2], data[0].shape[2]))

        # Image conditions covariance matrix of shape:
        # Image conditions × EEG channels × EEG channels
        sigma_cond = np.empty((data.shape[0], data.shape[2], data.shape[2]))
        
        for i in tqdm(range(data.shape[0])):
            cond_data = data[i]
            # Compute covariace matrices at each time point, and then
            # average across time points
            if self.__mvnn_dim == "time":
                sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                    shrinkage='auto') for t in range(cond_data.shape[2])],
                    axis=0)
            # Compute covariace matrices at each epoch (EEG repetition),
            # and then average across epochs/repetitions
            elif self.__mvnn_dim == "epochs":
                sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                    shrinkage='auto') for e in range(cond_data.shape[0])],
                    axis=0)
        # Average the covariance matrices across image conditions
        sigma = sigma_cond.mean(axis=0)

        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
        return sigma_inv

    def __whiten_data(self, data, sigma_inv):
        whitened_data = np.reshape((np.reshape(data, 
                (-1, data.shape[2], data.shape[3])).swapaxes(1, 2) 
                @ sigma_inv).swapaxes(1, 2), data.shape)
        return whitened_data