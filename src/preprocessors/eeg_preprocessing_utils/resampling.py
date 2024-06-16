from abc import ABC, abstractmethod
import mne

class Resampling(ABC):

    def __init__(self, **kwargs):
        pass    

    @abstractmethod
    def transform(self, data):
        pass

class ResamplingByFreq(Resampling):

    def __init__(self, sfreq=100, **kwargs):
        super().__init__(**kwargs)
        self.__sfreq = sfreq


    def transform(self, data):
        ch_names = data['ch_names']
        sfreq = data['sfreq']
        ch_types = data['ch_types']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        if self.__sfreq < sfreq:
            raw = mne.io.RawArray(data["raw_eeg_data"], info)
            del data["raw_eeg_data"]
            raw_resampled = raw.resample(self.__sfreq)
            data["raw_eeg_data"] = raw_resampled.get_data()
            data["sfreq"] = raw_resampled.info["sfreq"]
        return data

