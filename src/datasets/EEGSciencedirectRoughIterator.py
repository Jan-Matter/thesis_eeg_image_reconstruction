import torch
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import numpy as np
import os
import sys

load_dotenv(find_dotenv())
class EEGSciencedirectRoughIterator:

    def __init__(self):
        self.__data_path = str(Path(__file__).parent.parent.parent / os.getenv("SCIENCE_DIRECT_DATA_FOLDER"))
        self.__curr_itr_index = 0
        self.__load_data = True
        self.__file_paths = self.__init_file_paths(self.__data_path)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__curr_itr_index >= len(self.__file_paths):
            raise StopIteration
        else:
            file_path = self.__file_paths[self.__curr_itr_index]
            self.__curr_itr_index += 1
            print(f"{self.__curr_itr_index} of {len(self.__file_paths)} rough data file loaded")
            subj = int(file_path.split("/")[-3].split("-")[-1])
            session = int(file_path.split("/")[-2].split("-")[-1])
            split = file_path.split("/")[-1].split("_")[-1].split(".")[0]
            return {
                "eeg_data": self.get(subj, session, split) if self.load_data else None,
                "subj": subj,
                "session": session,
                "split": split
            }

    def get(self, subj, session, split):

        file_path = self.__data_path + f"/sub-{'0' if subj < 10 else ''}{subj}/ses-{'0' if session < 10 else ''}{session}/raw_eeg_{split}.npy"
        return {
            "file_path": file_path,
            "data": self.__get_data_from_file_path(file_path),
            "subj": subj,
            "session": session,
            "split": split
        }
    
    @property
    def file_paths(self):
        return self.__file_paths
    

    def subjs(self):
        subjs_set = {}
        for file_path in self.__file_paths:
            subjs_set = {*subjs_set, int(file_path.split("/")[-3].split("-")[-1])}
        subjs = list(subjs_set)
        subjs.sort()
        return subjs
    
    
    def session_count(self, subj):
        session_count = 0
        for file_path in self.__file_paths:
            if int(file_path.split("/")[-3].split("-")[-1]) == subj:
                session_count += 1
        return int(session_count / 2)
    
    @property
    def load_data(self):
        return self.__load_data
    
    @load_data.setter
    def load_data(self, value: bool):
        self.__load_data = value
        
    
    def __init_file_paths(self, data_path):
        file_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".npy") and not 'metadata' in file:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        file_paths.sort()
        return file_paths
    
    def __get_data_from_file_path(self, file_path):
        data = np.load(file_path, allow_pickle=True).item()
        return data
                    
                    
if __name__ == "__main__":
    dataset = EEGSciencedirectRoughIterator()
    data = dataset.get(1, 1, "training")
    print(data["data"].keys())
    print(data["data"]["raw_eeg_data"].shape)
    print("session_count", dataset.session_count(1))
    print("subjs", dataset.subjs())
    """
    for data in dataset:
        print(data["subj"])
        print(data["session"])
        print(data["split"])
        del data
    """
    