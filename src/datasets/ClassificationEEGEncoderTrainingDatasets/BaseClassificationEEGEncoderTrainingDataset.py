from typing_extensions import SupportsIndex
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np
from typing import Any, List, Union
import os
import cv2
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator

load_dotenv(find_dotenv())
class ClassificationEEGEncoderTrainingDataset(Dataset):

    def __init__(self,
                latent_dataset: Union[LatentImageSciencedirectDataset, LatentTextSciencedirectDataset],
                limit_to_classes: [List, None] = None,
                equalize_classes: bool = True
                ):
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__eeg_dataset = latent_dataset
        self.__limit_to_classes = limit_to_classes
        self.img_class_to_one_hot = self.__get_img_class_to_one_hot()
        self.img_concept_to_one_hot = self.__get_img_concept_to_one_hot()
        self.__idx_eeg_map = self.__init_idx_eeg_map()


    @property
    def num_classes(self) -> int:
        raise NotImplementedError("This method must be implemented by the child class")


    def __len__(self):
        return len(self.__idx_eeg_map)
    
    def __getitem__(self, idx):
        eeg_data = self.__idx_eeg_map[idx]
        split = eeg_data['split']
        img_condition = eeg_data['img_condition']
        img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
        img_class = next(self.__queryier.run_query(f"""SELECT image.img_class FROM image WHERE image.img_id = {img_id}"""))['img_class']
        img_concept = next(self.__queryier.run_query(f"""SELECT image.img_concept FROM image WHERE image.img_id = {img_id}"""))['img_concept']

        return {
            'eeg_data_delta': eeg_data['eeg_data_delta'],
            'eeg_data_theta': eeg_data['eeg_data_theta'],
            'eeg_data_alpha': eeg_data['eeg_data_alpha'],
            'eeg_data_beta': eeg_data['eeg_data_beta'],
            'eeg_data_gamma': eeg_data['eeg_data_gamma'],
            'latent': eeg_data['latent'],
            'img_class': self.img_class_to_one_hot[img_class],
            'img_concept': self.img_concept_to_one_hot[img_concept],
            'img_id': img_id,
            'img_condition': img_condition,
            'subj': eeg_data['subj'],
            'session': eeg_data['session'],
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

        # Define the dictionary
        one_hot_dict = {}
        for i, c in enumerate(classes):
            one_hot_dict[c] = [int(j == i) for j in range(len(classes))]

        # Convert the dictionary to torch tensors
        for k, v in one_hot_dict.items():
            one_hot_dict[k] = torch.tensor(v)

        return one_hot_dict
    
    def __get_img_concept_to_one_hot(self):

        where_clause = "" if self.__limit_to_classes is None else\
            f"""WHERE image.img_class IN ({','.join([f"'{c}'" for c in self.__limit_to_classes])})"""
        concepts = [concept['img_concept'] for concept in self.__queryier.run_query(f"""SELECT DISTINCT image.img_concept FROM image {where_clause}""")]
        
        # Define the dictionary
        one_hot_dict = {}
        for i, c in enumerate(concepts):
            one_hot_dict[c] = [int(j == i) for j in range(len(concepts))]
        
        # Convert the dictionary to torch tensors
        for k, v in one_hot_dict.items():
            one_hot_dict[k] = torch.tensor(v)
        
        return one_hot_dict
    

    def __init_idx_eeg_map(self):
        if self.__limit_to_classes is None:
            return [i for i in range(len(self.__eeg_dataset))]
        
        idx_eeg_map = []
        for i in range(len(self.__eeg_dataset)):
            eeg_data = self.__eeg_dataset[i]
            split = eeg_data['split']
            img_condition = eeg_data['img_condition']
            img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
            img_class = next(self.__queryier.run_query(f"""SELECT image.img_class FROM image WHERE image.img_id = {img_id}"""))['img_class']
            if img_class in self.__limit_to_classes:
                idx_eeg_map.append(eeg_data)
        return idx_eeg_map
    
    def __extract_foreground_image(self, img):
        mask = self.__extract_foreground_image_mask(img)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img
    
    def __extract_background_image(self, img):
        mask = self.__extract_foreground_image_mask(img)
        img = cv2.bitwise_not(img, img, mask=mask)
        return img
    

    def __extract_object_width(self, img):
        mask = self.__extract_foreground_image_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            width = w
        else:
            width = 0
        return width
    

    def __extract_object_height(self, img):
        mask = self.__extract_foreground_image_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            height = h
        else:
            height = 0
        return height
    

    def __extract_bounding_box_position(self, img):
        mask = self.__extract_foreground_image_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            position = (x, y)
        else:
            position = (0, 0)
        return position
    

    def __extract_foreground_image_mask(self, img):

        img = img[:, :, :3]  # Remove alpha channel
        img = img.astype('uint8')
        height, width = img.shape[:2]
        
        # Define the rectangle for prior
        prior_width = width * 3 // 4
        prior_height = height * 3 // 4
        prior_x = (width - prior_width) // 2
        prior_y = (height - prior_height) // 2
        prior_rect = (prior_x, prior_y, prior_width, prior_height)
        
        # Create a mask initialized with zeros
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # Set the foreground and background models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut algorithm
        cv2.grabCut(img, mask, prior_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create a mask where 0 and 2 are considered background, everything else is considered foreground
        mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
        
        return mask



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
    dataset = ClassificationEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=['toy']
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    for data in dataloader:
        print(data['eeg_data_delta'].shape)
        print(data['img_class'])
        print(data['img_condition'])
        print(data['img_concept'].shape)
        print(data['subj'])
        print(data['session'])
        print(data['split'])
        break