from abc import ABC, abstractmethod
from sklearn.utils import shuffle
import numpy as np
import mne

class Epoching(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass

class EpochingByMNE(Epoching):

    def __init__(self, tmin=-0.2, tmax=0.8, sfreq=250, baseline=(None, 0), **kwargs):
        super().__init__(**kwargs)
        self.__tmin = tmin
        self.__tmax = tmax
        self.__sfreq = sfreq
        self.__baseline = baseline

    def transform(self, data, max_img_rep, seed):
        epochs = self.__retrieve_epochs(data)
        sorted_epochs = self.__sort_epochs(epochs, data["sfreq"], max_img_rep, seed)
        return sorted_epochs

    def __retrieve_epochs(self, data):
        # Retrieve the EEG data and the events
        ch_types = data["ch_types"]
        ch_names = data["ch_names"]
        sfreq = data["sfreq"]

        # Create the info object needed by the MNE EpochsArray
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)    
        raw = mne.io.RawArray(data["raw_eeg_data"], info)
        del data["raw_eeg_data"]

        events = mne.find_events(raw, stim_channel='stim')
        
        #reject target trials events
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        # Create the MNE EpochsArray
        epochs = mne.Epochs(raw, events=events, tmin=self.__tmin, 
                        tmax=self.__tmax, baseline=self.__baseline, preload=True)
        
        epochs.resample(self.__sfreq)
        ch_names = epochs.info['ch_names']
        
        return epochs
    
    def __sort_epochs(self, epochs, sfreq, max_img_rep, seed):
        data = epochs.get_data()
        events = epochs.events[:,2]
        ch_names = epochs.info['ch_names']
        times = epochs.times

        
        img_cond = np.unique(events)

        del epochs

        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond), max_img_rep, data.shape[1] - 1,
            data.shape[2] - 50))
        
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            if len(idx) < 1:
                continue
            # Randomly select only the max number of EEG repetitions
            #idx = shuffle(idx, random_state=seed, n_samples=max_img_rep)
            idx = idx[:max_img_rep]
            sorted_data[i] = data[idx][:,:-1,50:] #remove the stim channel
        del data

        return{
            "epoched_eeg_data": sorted_data,
            "img_cond": img_cond,
            "ch_names": ch_names,
            "times": times,
            "sfreq": sfreq
        }