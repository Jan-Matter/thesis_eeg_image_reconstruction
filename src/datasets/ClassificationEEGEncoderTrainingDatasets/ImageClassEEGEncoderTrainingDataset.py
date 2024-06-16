from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset, DataLoader
from typing import Union
import torch
from pathlib import Path
import numpy as np
from typing import List
import os
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator

load_dotenv(find_dotenv())
class ImageClassEEGEncoderTrainingDataset(Dataset):

    def __init__(self,
                latent_dataset: Union[LatentImageSciencedirectDataset, LatentTextSciencedirectDataset],
                limit_to_classes: [List, None] = None,
                equalize_classes: bool = True
                ):
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__eeg_dataset = latent_dataset
        self.__limit_to_classes = limit_to_classes
        self.img_class_to_one_hot = self.__get_img_class_to_one_hot()
        self.__idx_eeg_map = self.__init_idx_eeg_map()
        if equalize_classes:
            self.__equalize_classes()


    @property
    def num_classes(self):
        return len(self.img_class_to_one_hot)
    

    def __len__(self):
        return len(self.__idx_eeg_map)
    
    def __getitem__(self, idx):
        eeg_data = self.__idx_eeg_map[idx]
        split = eeg_data['split']
        img_condition = eeg_data['img_condition']
        img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
        img_class = next(self.__queryier.run_query(f"""SELECT image.img_class FROM image WHERE image.img_id = {img_id}"""))['img_class']
        return {
            #'eeg_data_delta': eeg_data['eeg_data_delta'],
            #'eeg_data_theta': eeg_data['eeg_data_theta'],
            #'eeg_data_alpha': eeg_data['eeg_data_alpha'],
            #'eeg_data_beta': eeg_data['eeg_data_beta'],
            #'eeg_data_gamma': eeg_data['eeg_data_gamma'],
            'eeg_data': eeg_data['eeg_data'],
            'class': self.img_class_to_one_hot[img_class],
            'latent': eeg_data['latent'],
            'img_id': img_id,
            'img_condition': img_condition,
            'subj': eeg_data['subj'],
            'split': split
        }
    

    def __get_img_class_to_one_hot(self):
        classes = [
            'toy',
            'unknown',
            'food',
            'electronic device',
            'home decor',
            'vegetable',
            'musical instrument', 
            'weapon', 
            'animal', 
            'tool', 
            'clothing', 
            'sports equipment', 
            'vehicle', 
            'bird', 
            'clothing accessory', 
            'insect', 
            'furniture', 
            'container', 
            'medical equipment', 
            'kitchen tool', 
            'fruit', 
            'part of car', 
            'plant', 
            'drink', 
            'kitchen appliance', 
            'office supply', 
            'body part'
        ]
        
        classes = [c for c in classes if self.__limit_to_classes is None or c in self.__limit_to_classes]

        # Define the dictionary
        one_hot_dict = {}
        for i, c in enumerate(classes):
            one_hot_dict[c] = [int(j == i) for j in range(len(classes))]

        # Convert the dictionary to torch tensors
        for k, v in one_hot_dict.items():
            one_hot_dict[k] = torch.tensor(v)
        return one_hot_dict
    

    def __init_idx_eeg_map(self):
        
        idx_eeg_map = []
        for i in range(len(self.__eeg_dataset)):
            eeg_data = self.__eeg_dataset[i]
            split = eeg_data['split']
            img_condition = eeg_data['img_condition']
            img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
            img_class = next(self.__queryier.run_query(f"""SELECT image.img_class FROM image WHERE image.img_id = {img_id}"""))['img_class']
            if self.__limit_to_classes is None or img_class in self.__limit_to_classes:
                idx_eeg_map.append(eeg_data)
        return idx_eeg_map
    


    def __equalize_classes(self):
        class_counts = [0 for _ in range(self.num_classes)]
        idx_class_map = []
        for i in range(len(self)):
            eeg_data = self[i]
            curr_class = torch.argmax(eeg_data['class']).item()
            class_counts[curr_class] += 1
            idx_class_map.append(curr_class)
        
        min_class_count = np.min(class_counts)
        idx_eeg_map = []
        class_count = [0 for _ in range(self.num_classes)]
        for i in range(len(self)):
            curr_class = idx_class_map[i]
            if class_count[curr_class] < min_class_count:
                idx_eeg_map_val = self.__idx_eeg_map[i]
                idx_eeg_map.append(idx_eeg_map_val)
                class_count[curr_class] += 1
        self.__idx_eeg_map = idx_eeg_map
    


if __name__ == '__main__':
    data_loader = EEGSciencedirectRoughIterator() 
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="test",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )
    dataset = ImageClassEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=['animal', 'toy']
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    for data in dataloader:
        print(data['eeg_data_delta'].shape)
        print(data['img_condition'])
        print(data['class'])
        print(data['subj'])
        print(data['split'])
        break