from abc import ABC, abstractmethod
import numpy as np
import mne
mne.utils.set_log_level("ERROR")
import copy

class FrequencyFiltering(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass

class FrequencyBandFiltering(FrequencyFiltering):

    def __init__(self, frequency_bands, **kwargs):
        super().__init__(**kwargs)
        self.__frequency_band = frequency_bands

    def transform(self, data):
        time_series = {}
        time_series["bandpass"] = data["epoched_eeg_data"]
        sfreq = data["sfreq"]
        if "delta" in self.__frequency_band:
            time_series["delta"] = self.__filter_delta_band(data["epoched_eeg_data"], sfreq)
        if "theta" in self.__frequency_band:
            time_series["theta"] = self.__filter_theta_band(data["epoched_eeg_data"], sfreq)
        if "alpha" in self.__frequency_band:
            time_series["alpha"] = self.__filter_alpha_band(data["epoched_eeg_data"], sfreq)
        if "beta" in self.__frequency_band:
            time_series["beta"] = self.__filter_beta_band(data["epoched_eeg_data"], sfreq)
        if "gamma" in self.__frequency_band:
            time_series["gamma"] = self.__filter_gamma_band(data["epoched_eeg_data"], sfreq)
        data["time_series"] = time_series
        return data

    def __filter_delta_band(self, data, sfreq):
        data_cpy = data.copy()
        for img_cond_idx in range(data_cpy.shape[0]):
            data_cpy[img_cond_idx] = mne.filter.filter_data(data_cpy[img_cond_idx], sfreq=sfreq, l_freq=0.5, h_freq=4)
        return data_cpy
    
    def __filter_theta_band(self, data, sfreq):
        data_cpy = data.copy()
        for img_cond_idx in range(data_cpy.shape[0]):
            data_cpy[img_cond_idx] = mne.filter.filter_data(data_cpy[img_cond_idx], sfreq=sfreq, l_freq=4, h_freq=8)
        return data_cpy
    
    def __filter_alpha_band(self, data, sfreq):
        data_cpy = data.copy()
        for img_cond_idx in range(data_cpy.shape[0]):
            data_cpy[img_cond_idx] = mne.filter.filter_data(data_cpy[img_cond_idx], sfreq=sfreq, l_freq=8, h_freq=13)
        return data_cpy
    
    def __filter_beta_band(self, data, sfreq):
        data_cpy = data.copy()
        for img_cond_idx in range(data_cpy.shape[0]):
            data_cpy[img_cond_idx] = mne.filter.filter_data(data_cpy[img_cond_idx], sfreq=sfreq, l_freq=13, h_freq=30)
        return data_cpy
    
    def __filter_gamma_band(self, data, sfreq):
        data_cpy = data.copy()
        for img_cond_idx in range(data_cpy.shape[0]):
            data_cpy[img_cond_idx] = mne.filter.filter_data(data_cpy[img_cond_idx], sfreq=sfreq, l_freq=30, h_freq=49)
        return data_cpy
