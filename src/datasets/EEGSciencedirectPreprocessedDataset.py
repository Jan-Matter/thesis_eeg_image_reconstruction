from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List
import os
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

#only for testing can be later removed
#-----------------
from src.preprocessors.eeg_preprocessor import EEGPreprocessorBuilder
from src.preprocessors.eeg_preprocessing_utils.channel_selection import ChannelSelectionByName
from src.preprocessors.eeg_preprocessing_utils.resampling import ResamplingByFreq
from src.preprocessors.eeg_preprocessing_utils.epoching import EpochingByMNE
from src.preprocessors.eeg_preprocessing_utils.denoising import MVNN
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.preprocessors.eeg_preprocessing_utils.frequency_filtering import FrequencyBandFiltering
from src.preprocessors.eeg_preprocessing_utils.transforms import FourierTransform, MorletWaveletTransform
#-----------------

load_dotenv(find_dotenv())
class EEGSciencedirectPreprocessedDataset(Dataset):

    def __init__(self, rough_data_loader,
                 limit_to_split=None,
                 limit_to_subj: [List, None]=None,
                 avg_repetitions=False,              
                 overwrite_cache=False,
                 in_memory_cache_behaviour="last",
                 preprocess=False):
        self.__data_path = str(Path(__file__).parent.parent.parent / os.getenv("SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER_TRAINING_DATASET_NICE_ADAPTED"))

        if not avg_repetitions:
            raise ValueError("This configuration is not fully supported yet!")

        self.__rough_data_loader = rough_data_loader
        self.__avg_repetitions = avg_repetitions
        self.__freq_band_to_idx = {
            'delta': 0,
            'theta': 1,
            'alpha': 2,
            'beta': 3,
            'gamma': 4
        }
        self.__overwrite_cache = overwrite_cache
        self.__in_memory_cache_behaviour = in_memory_cache_behaviour
        self.__preprocess_configs_train = {"max_img_rep": 2, "seed": 0}
        self.__preprocess_configs_test = {"max_img_rep": 20, "seed": 0}
        self.__in_memory_cache = {}
        self.__curr_index = 0

        self.__search_index = self.__init_index(
            limit_to_split=limit_to_split,
            limit_to_subj=limit_to_subj
        )


    @property
    def preprocess_configs_train(self):
        return self.__preprocess_configs_train

    @preprocess_configs_train.setter
    def preprocess_configs_train(self, value):
        self.__preprocess_configs_train = value

    @property
    def preprocess_configs_test(self):
        return self.__preprocess_configs_test

    @preprocess_configs_test.setter
    def preprocess_configs_test(self, value):
        self.__preprocess_configs_test = value

    @property
    def overwrite_cache(self):
        return self.__overwrite_cache

    @overwrite_cache.setter
    def overwrite_cache(self, value):
        self.__overwrite_cache = value
    
    def __len__(self):
        max_index = max(self.__search_index.keys())
        return max_index + self.__search_index[max_index]["length"]
    
    def __getitem__(self, index):
        file_dict = self.__index_to_file_dict(index)
        img_condition =  index - file_dict["start_index"] + 1
        eeg_data = self.__get_data_from_file_path(file_dict["file_path"], img_condition)
        return {
            "file_path": file_dict["file_path"],
            "subj": file_dict["subj"],
            "split": file_dict["split"],
            "img_condition": img_condition,
            "eeg_data": eeg_data,
            #"eeg_data_delta": eeg_data[self.__freq_band_to_idx["delta"]],
            #"eeg_data_theta": eeg_data[self.__freq_band_to_idx["theta"]],
            #"eeg_data_alpha": eeg_data[self.__freq_band_to_idx["alpha"]],
            #"eeg_data_beta": eeg_data[self.__freq_band_to_idx["beta"]],
            #"eeg_data_gamma": eeg_data[self.__freq_band_to_idx["gamma"]]
                   
        }
    
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                next_elem = self[self.__curr_index]
                break
            except Exception as e:
                print(e)
                self.__curr_index += 1
        self.__curr_index += 1
        return next_elem
    
    def subjs(self):
        return self.__rough_data_loader.subjs()
    
    def session_count(self, subj):
        return self.__rough_data_loader.session_count(subj)
    
    def set_iterator_start_point(self, subj, session, split, img_condition):
        for i, curr in self.__search_index.items():
            for j, curr_session in curr.items():
                for k, curr_split in curr_session.items():
                    if curr_split["subj"] == subj and curr_split["session"] == session and curr_split["split"] == split:
                        self.__curr_index = k + img_condition - 1
                        return
                    
    
    def __init_index(self, limit_to_split=None, limit_to_subj=None):
        file_paths = sorted(os.listdir(self.__data_path), key=lambda x: (int(x.split("-")[1].split("_")[0]), x.split("-")[2].split(".")[0]))

        file_dict_lengths = [
            {
                "subj": int(path.split("-")[1].split("_")[0]),
                "split": path.split("-")[2].split(".")[0],
                "file_path": path
            }
            for path in file_paths
        ]
        
        for i, file_dict in enumerate(file_dict_lengths):
            if limit_to_subj is not None and file_dict["subj"] not in limit_to_subj:
                continue
            if limit_to_split is not None and file_dict["split"] not in limit_to_split:
                continue
            data = np.load(self.__data_path + "/"  + file_dict["file_path"])
            file_dict["length"] = data.shape[0]
            file_dict_lengths[i] = file_dict

        curr_index = 0
        search_index = {}

                
        for i, file_dict in enumerate(file_dict_lengths):
            if limit_to_subj is not None and file_dict["subj"] not in limit_to_subj:
                continue
            if limit_to_split is not None and file_dict["split"] not in limit_to_split:
                continue
            search_index[curr_index] = file_dict
            search_index[curr_index]["start_index"] = curr_index
            curr_index += file_dict["length"]

        return search_index
    
    
    def __get_data_from_file_path(self, file_path, img_condition):

        if file_path in self.__in_memory_cache:
            cached_condition_data = self.__in_memory_cache[file_path][img_condition - 1]
            return cached_condition_data
        
        data = np.load(self.__data_path + "/"  + file_path)
        #since cache files can be very large only one data file is loaded to cache
        if self.__in_memory_cache_behaviour != "all":
            del self.__in_memory_cache
            self.__in_memory_cache = {}
        
        if self.__in_memory_cache_behaviour == "last" or self.__in_memory_cache_behaviour == "all":
            self.__in_memory_cache[file_path] = data

        condition_data = data[img_condition - 1]
        return condition_data
    
    def __index_to_file_dict(self, index):
        indexes = sorted(list(self.__search_index.keys()), reverse=True)
        #iterate from highest to lowest index
        for i in indexes:
            if index >= i:
                return self.__search_index[i]
                    
                    
if __name__ == "__main__":
    data_loader = EEGSciencedirectRoughIterator()  
    dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
                                                    limit_to_subj=[1],
                                                    limit_to_split="training",
                                                    avg_repetitions=True,
                                                    in_memory_cache_behaviour="last",
                                                    overwrite_cache=False,
                                                    preprocess=True)
    

    for i, data in enumerate(dataset):
        print(data["subj"])
        print(data["split"])
        print(data["img_condition"])
        print(data["eeg_data"].shape)
        
        
    