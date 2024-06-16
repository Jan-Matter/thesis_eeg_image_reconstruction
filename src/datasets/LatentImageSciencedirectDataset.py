from dotenv import load_dotenv, find_dotenv
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import os
import sys
from einops import repeat, rearrange
from omegaconf import OmegaConf
from PIL import Image
import PIL
import torch
from torch.functional import F
import torchvision


load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.stablediffusion.util import instantiate_from_config
from src.stablediffusion.data.util import AddMiDaS

class LatentImageSciencedirectDataset(Dataset):

    def __init__(self, gpu=0, load_model=True, mode="latent_diff", downsample_size=(16, 16)):
        self.__data_path = Path(os.getenv("SCIENCE_DIRECT_LATENT_DATA_FOLDER"))
        self.__config_path = Path(os.getenv("CONFIG_FOLDER")) / "inference/eeg_to_image_decoder.yaml"
        self.__config = OmegaConf.load(self.__config_path)['inference']
        self.__model_path = Path(os.getenv("MODEL_FOLDER")) / "sd-v1-4.ckpt"
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        self.__mode = mode
        self.__downsample_size = downsample_size
        self.__gpu = gpu
        if self.__gpu > 0:
            torch.cuda.set_device(self.__gpu)
        self.__device = torch.device(f"cuda:{self.__gpu}") if torch.cuda.is_available() else torch.device("cpu")
        if load_model:
            self.__model = self.__initialize_model(self.__config['model'], self.__config['ckpt'], self.__device)
        else:
            self.__model = None
        self.__in_memory_cache = {}
        self.__fill_in_memory_cache()

    
    def __len__(self):
        return next(self.__queryier.run_query(f"""SELECT COUNT(DISTINCT(image.img_id)) FROM image"""))['COUNT(DISTINCT(image.img_id))']
    
    def __getitem__(self, idx):
        return self.get_latent_img(idx)

    def get_latent_img(self, img_id, use_cached=True):
        if self.__mode == "downsample":
            return self.__get_downsampled_img(img_id, use_cached)
        elif self.__mode == "latent_diff":
            if use_cached and self.__has_cached_latent_img(img_id):
                return self.__load_cached_latent_img(img_id)
            else:
                return self.__encode_and_cache_img(img_id)
        else:
            raise ValueError(f"Invalid mode {self.mode}")
    
    def encode_img(self, img):
        
        init_image = Image.fromarray(img.astype(np.uint8)).resize((512, 512))
        image = self.__load_image(init_image)
        #image = self.__pad_image(init_image)
        #image = self.__transform_image(image, self.__device)

        # resize image from (1, 3, 512, 512) to (1, 3, 128, 128)
        image_resized = F.interpolate(image, size=(128, 128), mode='bilinear')
        
        with torch.no_grad():
            z = self.__model.get_first_stage_encoding(self.__model.encode_first_stage(image_resized))

        return z
    
    def __initialize_model(self, config, ckpt, gpu, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        model.to(gpu)
        model.eval()
        return model
    
    def __load_image(self, image):
        image = image.convert("RGB")
        w, h = image.size
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to('cuda')
        return 2. * image - 1.
    
    def __has_cached_latent_img(self, img_id):
        if img_id in self.__in_memory_cache:
            return True
        path = self.__construct_latent_img_path(img_id)
        return path.exists()
    
    def __construct_latent_img_path(self, img_id):
        if self.__mode == "downsample":
            return self.__data_path / "downsampled_img" / f"{img_id}.npy"
        return self.__data_path / "img" / f"{img_id}.npy"

    def __load_cached_latent_img(self, img_id):
        if img_id in self.__in_memory_cache:
            return self.__in_memory_cache[img_id]
        path = self.__construct_latent_img_path(img_id)
        latent_img = np.load(path)
        return torch.from_numpy(latent_img)
    
    def __save_latent_img(self, img_id, latent_img):
        path = self.__construct_latent_img_path(img_id)
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, latent_img.numpy())
    
    def __encode_and_cache_img(self, img_id):
        img = next(self.__queryier.run_query(f"""SELECT image.img_path FROM image WHERE image.img_id = {img_id}"""))['img']
        latent_img = self.encode_img(img).cpu()
        self.__save_latent_img(img_id, latent_img)
        return latent_img
    
    def __get_downsampled_img(self, img_id, use_cached=True):
        img = next(self.__queryier.run_query(f"""SELECT image.img_path FROM image WHERE image.img_id = {img_id}"""))['img']
        img = np.mean(img, axis=2)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        latent_img = F.interpolate(img, size=self.__downsample_size, mode='bilinear', align_corners=False)
        return latent_img
    
    def __fill_in_memory_cache(self):
        for img_id in range(1, len(self) + 1):
            latent_img = self.get_latent_img(img_id, use_cached=True)
            self.__in_memory_cache[img_id] = latent_img

if __name__ == "__main__":
    dataset = LatentImageSciencedirectDataset(gpu=0, load_model=True, mode="latent_diff", downsample_size=(16, 16))
    #dataset = LatentImageSciencedirectDataset(gpu=0, load_model=True, mode="latent_diff")
    dataset.get_latent_img(3, use_cached=True)
    for i in range(len(dataset)):
        img_latent = dataset[i + 1]
        print(img_latent.shape)
