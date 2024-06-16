from abc import ABC, abstractmethod
import mne
import numpy as np
import re

class ChannelSelection(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass


class ChannelSelectionByName(ChannelSelection):

    def __init__(self, names=[], regex=None, **kwargs):
        super().__init__(**kwargs)
        self.__names = names
        self.__regex = regex

    def transform(self, data):
        if len(self.__names) > 0:
            return self.__select_by_names(data)
        elif self.__regex is not None:
            return self.__select_by_regex(data)
        else:
            return data
    
    def __select_by_regex(self, data):
        regex = re.compile(self.__regex)
        ch_names = data["ch_names"]
        regex_matches = [ch_name for ch_name in ch_names if regex.match(ch_name)]
        if not regex_matches:
            raise ValueError(f'No channels found matching regex pattern {self.__regex}!')
        channel_idx = mne.pick_channels(data["ch_names"], include=regex_matches)
        stim_idx = mne.pick_channels(data["ch_names"], include=["stim"])
        channel_idx = np.append(channel_idx, stim_idx)
        data["ch_names"] = [ch_names[i] for i in channel_idx]
        data["ch_types"] = [data["ch_types"][i] for i in channel_idx]
        data["raw_eeg_data"] = data["raw_eeg_data"][channel_idx, :]
        return data
    
    def __select_by_names(self, data):
        #check if all names are in data
        for name in self.__names:
            if name not in data["ch_names"]:
                raise ValueError(f'Channel {name} not found in data!')
        channel_idx = mne.pick_channels(data["ch_names"], include=self.__names, ordered=True)
        stim_idx = mne.pick_channels(data["ch_names"], include=["stim"])
        channel_idx = np.append(channel_idx, stim_idx)
        data["ch_names"] = [data["ch_names"][i] for i in channel_idx]
        data["ch_types"] = [data["ch_types"][i] for i in channel_idx]
        data["raw_eeg_data"] = data["raw_eeg_data"][channel_idx, :]
        return data
        
