from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os
import sys
from einops import repeat, rearrange
from omegaconf import OmegaConf
import torch
from PIL import Image
from torch.utils.data.dataset import ConcatDataset
from torch.functional import F

import open_clip


load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.stablediffusion.util import instantiate_from_config
from src.stablediffusion.data.util import AddMiDaS

class LatentTextSciencedirectDataset(Dataset):

    def __init__(self, load_model=True, gpu=0):
        self.__data_path = Path(os.getenv("SCIENCE_DIRECT_LATENT_DATA_FOLDER"))
        self.__config_path = Path(os.getenv("CONFIG_FOLDER")) / "encoding_text.yaml"
        self.__model_path = Path(os.getenv("MODEL_FOLDER")) / "sd-v1-4.ckpt"
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__gpu = gpu
        self.__in_memory_cache = {}
        self.__device = torch.device(f"cuda:{self.__gpu}") if torch.cuda.is_available() else torch.device("cpu")
        if load_model:
            self.model, self.preprocess = self.__initialize_model(self.__config_path, self.__model_path, self.__device)
        self.__fill_in_memory_cache()
    
    def __len__(self):
        return next(self.__queryier.run_query(f"""SELECT COUNT(DISTINCT(image.img_id)) FROM image"""))['COUNT(DISTINCT(image.img_id))']
    
    def __getitem__(self, idx):
        idx = idx + 1
        return self.get_latent_img(idx)

    def get_latent_img(self, img_id, use_cached=True):
        if use_cached and img_id in self.__in_memory_cache:
            return self.__in_memory_cache[img_id]
        if use_cached and self.__has_cached_latent_img(img_id):
            return self.__load_cached_latent_img(img_id)
        else:
            return self.__encode_and_cache_img(img_id)
    
    def encode_img(self, img):
        if self.__gpu > 0:
            torch.cuda.set_device(self.__gpu)

        image = self.preprocess(Image.fromarray(img.astype(np.uint8)).resize((512,512))).unsqueeze(0).to(self.__device)
        image_latent = self.model.encode_image(image)
        return image_latent
        
    
    def __initialize_model(self, config_path, model_path, device):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L/14', pretrained='laion2b_s32b_b82k')
        model = model.to(device)
        return model, preprocess
    
    def __has_cached_latent_img(self, img_id):
        if img_id in self.__in_memory_cache:
            return True
        path = self.__construct_latent_img_caption_path(img_id)
        return path.exists()
    
    def __construct_latent_img_caption_path(self, img_id):
        return self.__data_path / "semantic" / f"{img_id}.npy"

    def __load_cached_latent_img(self, img_id):
        if img_id in self.__in_memory_cache:
            return self.__in_memory_cache[img_id]
        path = self.__construct_latent_img_caption_path(img_id)
        latent_text = np.load(path)
        return torch.from_numpy(latent_text)
    
    def __save_latent_img(self, img_id, latent_img):
        path = self.__construct_latent_img_caption_path(img_id)
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, latent_img.numpy())
    
    def __encode_and_cache_img(self, img_id):
        img = next(self.__queryier.run_query(f"""SELECT image.img_path FROM image WHERE image.img_id = {img_id}"""))['img']
        latent_image = self.encode_img(img).cpu().detach()
        self.__save_latent_img(img_id, latent_image)
        return latent_image

    def __fill_in_memory_cache(self):
        for img_id in range(1, len(self) + 1):
            latent_image = self.get_latent_img(img_id, use_cached=True)
            self.__in_memory_cache[img_id] = latent_image



if __name__ == "__main__":
    dataset = LatentTextSciencedirectDataset(gpu=0)
    dataset.get_latent_img(1, use_cached=False)
    