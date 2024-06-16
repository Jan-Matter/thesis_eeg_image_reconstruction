from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np
from typing import List, Union
import os
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator

load_dotenv(find_dotenv())
class ImageConceptEEGEncoderTrainingDataset(Dataset):

    def __init__(self,
                latent_dataset: Union[LatentImageSciencedirectDataset, LatentTextSciencedirectDataset],
                limit_to_classes: [List, None] = None,
                limit_to_concepts: [List, None] = None,
                limit_to_split: [str, None] = None,
                equalize_classes: bool = True
                ):
        self.__limit_to_split = limit_to_split
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__eeg_dataset = latent_dataset
        self.__limit_to_classes = limit_to_classes
        self.__limit_to_concepts = limit_to_concepts
        self.__valid_concept_ids = None
        #if self.__limit_to_classes is None and self.__limit_to_concepts is not None:
        #    raise ValueError("Cannot limit to concepts without limiting to classes")
        self.__idx_eeg_map = self.__init_idx_eeg_map()
        self.img_concept_to_one_hot = self.__get_img_concept_to_one_hot()
        if equalize_classes:
            self.__equalize_classes()


    @property
    def num_classes(self):
        return len(self.img_concept_to_one_hot)
    
    @property
    def concept_ids(self):
        return self.__valid_concept_ids


    def __len__(self):
        return len(self.__idx_eeg_map)
    
    def __getitem__(self, idx):
        eeg_data = self.__idx_eeg_map[idx]
        split = eeg_data['split']
        img_condition = eeg_data['img_condition']
        img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
        img_concept_id = next(self.__queryier.run_query(f"""SELECT image.img_concept_id FROM image WHERE image.img_id = {img_id}"""))['img_concept_id']
        return {
            #'eeg_data_delta': eeg_data['eeg_data_delta'],
            #'eeg_data_theta': eeg_data['eeg_data_theta'],
            #'eeg_data_alpha': eeg_data['eeg_data_alpha'],
            #'eeg_data_beta': eeg_data['eeg_data_beta'],
            #'eeg_data_gamma': eeg_data['eeg_data_gamma'],
            'eeg_data': eeg_data['eeg_data'],
            'latent': eeg_data['latent'],
            'class': self.img_concept_to_one_hot[img_concept_id],
            'img_id': img_id,
            'img_condition': img_condition,
            'subj': eeg_data['subj'],
            'split': split
        }

    
    def __get_img_concept_to_one_hot(self):

        #where_clause = "" if self.__limit_to_classes is None else\
        #    f"""WHERE {"image.split = '" + self.__limit_to_split + "' AND" if self.__limit_to_split is not None else ""} image.img_class IN ({','.join([f"'{c}'" for c in self.__limit_to_classes])})
        #    """
        #concepts = [concept['img_concept_id'] for concept in self.__queryier.run_query(f"""SELECT DISTINCT image.img_concept_id FROM image {where_clause}""")]

        #if not self.__limit_to_concepts is None:
        #    concepts = [concepts[c] for c in self.__limit_to_concepts]
        concepts = self.__valid_concept_ids

        # Define the dictionary
        one_hot_dict = {}
        for i, c in enumerate(concepts):
            one_hot_dict[c] = [int(j == i) for j in range(len(concepts))]
        
        # Convert the dictionary to torch tensors
        for k, v in one_hot_dict.items():
            one_hot_dict[k] = torch.tensor(v)
        
        return one_hot_dict
    

    def __init_idx_eeg_map(self):
        
        #where_clause = "" if self.__limit_to_classes is None else\
        #    f"""WHERE {"image.split = '" + self.__limit_to_split + "' AND" if self.__limit_to_split else ""} image.img_class IN ({','.join([f"'{c}'" for c in self.__limit_to_classes])})"""
        #concepts = [concept['img_concept_id'] for concept in self.__queryier.run_query(f"""SELECT DISTINCT image.img_concept_id FROM image {where_clause}""")]
        

        #if not self.__limit_to_concepts is None:
        #    concepts = [concepts[c] for c in self.__limit_to_concepts]
        
        concepts = []

        idx_eeg_map = []
        for i in range(len(self.__eeg_dataset)):
            eeg_data = self.__eeg_dataset[i]
            split = eeg_data['split']
            img_condition = eeg_data['img_condition']
            img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
            img_class = next(self.__queryier.run_query(f"""SELECT image.img_class FROM image WHERE image.img_id = {img_id}"""))['img_class']
            img_concept_id = next(self.__queryier.run_query(f"""SELECT image.img_concept_id FROM image WHERE image.img_id = {img_id}"""))['img_concept_id']
            if self.__limit_to_classes is None or img_class in self.__limit_to_classes:
                if img_concept_id not in concepts:
                    concepts.append(img_concept_id)
                self.__valid_concept_ids = [concept for i, concept in enumerate(concepts) if self.__limit_to_concepts is None or i in self.__limit_to_concepts]
                if img_concept_id in self.__valid_concept_ids:
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
    dataset = ImageConceptEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=['animal'],
        limit_to_concepts=[1, 2]
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    for data in dataloader:
        print(data['eeg_data_delta'].shape)
        print(data['img_condition'])
        print(data['class'])
        print(data['subj'])
        print(data['split'])
        break