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
class ForegroundColorEEGEncoderTrainingDataset(Dataset):

    def __init__(self,
                latent_dataset: Union[LatentImageSciencedirectDataset, LatentTextSciencedirectDataset],
                limit_to_classes: [List, None] = None,
                sampling_freq: int = 10,
                equalize_classes: bool = True
                ):
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__eeg_dataset = latent_dataset
        self.__limit_to_classes = limit_to_classes
        self.__idx_eeg_map = self.__init_idx_eeg_map()
        self.__color_class_map = {
            'blue': torch.tensor([1, 0, 0]),
            'green': torch.tensor([0, 1, 0]),
            'brown': torch.tensor([0, 0, 1]),
        }
        self.__sampling_freq = sampling_freq
        self.__cache_folder = Path('data/classification_cache/foregroundcolor_cache')
        os.makedirs(self.__cache_folder, exist_ok=True)
        self.__in_memory_cache = {}
        if equalize_classes:
            self.__equalize_classes()

    @property
    def num_classes(self):
        return len(self.__color_class_map)

    def __len__(self):
        return len(self.__idx_eeg_map)
    

    def __getitem__(self, idx):
        eeg_data = self.__idx_eeg_map[idx]
        split = eeg_data['split']
        img_condition = eeg_data['img_condition']
        img_id = next(self.__queryier.run_query(f"""SELECT image.img_id FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))['img_id']
        cache_path = self.__cache_folder / f"{img_id}.json"
        if cache_path.exists():
            color, intensity = self.__load_from_cache(cache_path)
        else:
            img_data = next(self.__queryier.run_query(f"""SELECT image.img_path FROM image WHERE image.split = '{split}' AND image.img_condition = {img_condition}"""))
            img = img_data['img']
            foreground_img = self.__extract_foreground_image(img)
            color, intensity = self.__extract_predominant_color_from_image(foreground_img)
            self.__save_to_cache(cache_path, color, intensity)
        
        return {
            #'eeg_data_delta': eeg_data['eeg_data_delta'],
            #'eeg_data_theta': eeg_data['eeg_data_theta'],
            #'eeg_data_alpha': eeg_data['eeg_data_alpha'],
            #'eeg_data_beta': eeg_data['eeg_data_beta'],
            #'eeg_data_gamma': eeg_data['eeg_data_gamma'],
            'eeg_data': eeg_data['eeg_data'],
            'class': self.__color_class_map[color],
            'value': intensity,
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
        
         
    
    def __extract_predominant_color_from_image(self, img):
        
        # Convert the image to the HSV color space
        img = img.astype('uint8')
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the color ranges for blue, green, red, gray, and yellow
        color_ranges = {
            'blue': ([100, 100, 50], [120, 255, 255]),
            'green': ([30, 20, 20], [90, 255, 255]),
            'brown': ([10, 70, 70], [20, 255, 255]),
        }

        # Initialize a dictionary to store the color counts
        color_counts = {color: 0 for color in color_ranges}

        # Iterate over each pixel in the image
        intensity = 0
        for i, row in enumerate(hsv_img):
            if i % self.__sampling_freq != 0:
                continue
            for j, pixel in enumerate(row):
                if j % self.__sampling_freq != 0:
                    continue
                # Check if the pixel falls within any of the color ranges
                for color, (lower, upper) in color_ranges.items():
                    if np.all(pixel >= lower) and np.all(pixel <= upper):
                        # Increment the count for the corresponding color
                        color_counts[color] += 1
                        #running intensity average
                        intensity += pixel[2]
                        intensity //= 2

        # Find the color with the maximum count
        predominant_color = max(color_counts, key=color_counts.get)

        return predominant_color, intensity
    
    def __extract_foreground_image(self, img):
        mask = self.__extract_foreground_image_mask(img)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img
    

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
        self.__in_memory_cache[cache_path] = data['color'], data['intensity']
        return data['color'], data['intensity']
    
    
    def __save_to_cache(self, cache_path, color, intensity):
        with open(cache_path, 'w') as f:
            json.dump({
                'color': color,
                'intensity': int(intensity)
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
    dataset = ForegroundColorEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=['vehicle']
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