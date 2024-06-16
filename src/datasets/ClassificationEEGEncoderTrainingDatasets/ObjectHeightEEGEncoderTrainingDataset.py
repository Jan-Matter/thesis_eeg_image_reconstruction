from typing_extensions import SupportsIndex
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np
from typing import Any, List, Union
import os
import cv2
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator

load_dotenv(find_dotenv())
class ObjectHeightEEGEncoderTrainingDataset(Dataset):

    def __init__(self,
                latent_dataset: Union[LatentImageSciencedirectDataset, LatentTextSciencedirectDataset],
                limit_to_classes: [List, None] = None,
                equalize_classes: bool = True
                ):
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__eeg_dataset = latent_dataset
        self.__limit_to_classes = limit_to_classes
        self.__idx_eeg_map = self.__init_idx_eeg_map()
        self.__heigth_to_class_map = {
            'unknown': torch.tensor([0, 0, 0]),
            'small': torch.tensor([1, 0, 0]),
            'medium': torch.tensor([0, 1, 0]),
            'large': torch.tensor([0, 0, 1]),
        }
        self.__cache_folder = Path('data/classification_cache/object_height_cache')
        os.makedirs(self.__cache_folder, exist_ok=True)
        self.__in_memory_cache = {}
        if equalize_classes:
            self.__equalize_classes()


    @property
    def num_classes(self):
        return len(self.__heigth_to_class_map) - 1


    def __len__(self):
        return len(self.__idx_eeg_map)
    

    def __getitem__(self, idx):
        eeg_data = self.__idx_eeg_map[idx]
        split = eeg_data['split']
        img_condition = eeg_data['img_condition']
        img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
        cache_path = self.__cache_folder / f"{img_id}.json"
        if cache_path.exists():
            height, height_class = self.__load_from_cache(cache_path)
        else:
            img_data = next(self.__queryier.run_query(f"""SELECT image.img_path FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))
            img = img_data['img']
            foreground_img = self.__extract_foreground_image(img)
            height = self.__extract_object_height(foreground_img)
            height_class = self.__heigth_to_class(height)
            self.__save_to_cache(cache_path, height, height_class)

        return {
            #'eeg_data_delta': eeg_data['eeg_data_delta'],
            #'eeg_data_theta': eeg_data['eeg_data_theta'],
            #'eeg_data_alpha': eeg_data['eeg_data_alpha'],
            #'eeg_data_beta': eeg_data['eeg_data_beta'],
            #'eeg_data_gamma': eeg_data['eeg_data_gamma'],
            'eeg_data': eeg_data['eeg_data'],
            'class': self.__heigth_to_class_map[height_class],
            'value': height,
            'latent': eeg_data['latent'],
            'img_id': img_id,
            'img_condition': img_condition,
            'subj': eeg_data['subj'],
            'split': split
        }
    

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
         
    
    
    
    def __extract_foreground_image(self, img):
        mask = self.__extract_foreground_image_mask(img)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img
    

    def __heigth_to_class(self, height):
        if height < 20:
            return 'unknown'
        if height < 100:
            return 'small'
        elif height < 300:
            return 'medium'
        else:
            return 'large'
    

    def __extract_object_height(self, img):
        mask = self.__extract_foreground_image_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            height = h
        else:
            height = 0
        return height
    

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
    

    def __load_from_cache(self, cache_path):
        if cache_path in self.__in_memory_cache:
            return self.__in_memory_cache[cache_path]
        with open(cache_path, 'r') as f:
            data = json.load(f)
        self.__in_memory_cache[cache_path] = data['height'], data['height_class']
        return data['height'], data['height_class']
    
    
    def __save_to_cache(self, cache_path, height, height_class):
        with open(cache_path, 'w') as f:
            json.dump({
                'height': height,
                'height_class': height_class
            }, f)



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
    dataset = ObjectHeightEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=['toy']
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    for data in dataloader:
        print(data['eeg_data_delta'].shape)
        print(data['class'])
        print(data['img_condition'])
        print(data['subj'])
        print(data['session'])
        print(data['split'])
        break