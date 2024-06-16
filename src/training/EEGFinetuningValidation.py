import sys
import os
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim import Adam
#from torch.utils.tensorboard import SummaryWriter
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.datasets.VisualEEGEncoderTrainingDataset import VisualEEGEncoderTrainingDataset
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.eeg_encoders.models.eeg_channel_net import EEChannelNet
from src.eeg_encoders.models.visual_eegnetv4_classifier import VisualEEGNetV4Classifier

class EEGEncoderValidation:

    def __init__(self, model, dataset, conf, limit_to_idxs):
        idxs = np.arange(len(dataset)) if limit_to_idxs is None else limit_to_idxs
        self.__batch_size = conf["batch_size"]
        self.__model_save_path = conf["model_save_path"]
        self.__all_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']
        self.__channels_used = np.array([i for i, channel in enumerate(self.__all_channels) if channel in conf["limit_to_channels"]])
        self.__freq_bands_used = conf["limit_to_freq_bands"]

        tensorboard_folder = os.getenv("TENSORBOARD_FOLDER") + f"/visual_eeg_encoder_v{conf['model_version']}"
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        #self.__writer = SummaryWriter(os.getenv("TENSORBOARD_FOLDER") + f"/visual_eeg_encoder_v{conf['model_version']}")

        if conf["gpu"] > 0:
            torch.cuda.set_device(conf["gpu"])
        self.__device = torch.device(f'cuda:{conf["gpu"]}') if torch.cuda.is_available() else torch.device("cpu")

        dataset = Subset(dataset, idxs)
        self.dataset = dataset
        self.validation_loader = DataLoader(dataset, batch_size=self.__batch_size, shuffle=False)
        self.__model = model.to(self.__device)
        self.__loss = nn.MSELoss()

    
    def validate(self):
        with torch.no_grad():
            self.__model.eval()
            losses = []
            for batch in self.validation_loader:
                batch = self.__filter_batch(batch)
                latent = batch["latent"].to(self.__device)
                output = self.__model(batch["eeg_data"])
                latent_image_flattened = torch.flatten(latent, start_dim=1)
                loss = self.__loss(output, latent_image_flattened)
                losses.append(loss)
            losses_tensor = torch.stack(losses)
            mean_loss = torch.mean(losses_tensor)
            #self.__writer.add_scalar('Loss/validation', mean_train_loss, epoch)
        return mean_loss

    
    def load_model(self, model_save_path):
        self.__model.load_state_dict(torch.load(model_save_path))
        self.__model.eval()

    
    def __filter_batch(self, batch):
        outbatch = batch.copy()
        
        outbatch["eeg_data"] = outbatch["eeg_data"][:, :, self.__channels_used, :]
        return outbatch


if __name__ == '__main__':
    validation_conf_path = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_to_latent_image_validation.yaml"
    validation_conf = yaml.load(validation_conf_path.open(), Loader=yaml.FullLoader)['validation']
    model_conf_path = Path(os.getenv("CONFIG_FOLDER"))/  "model" / "visual" / f"eeg_channel_net.yaml"
    model_conf = yaml.load(model_conf_path.open(), Loader=yaml.FullLoader)

    validation_conf["model_save_path"] = os.getenv("MODEL_FOLDER") + f"/eeg_channel_net_visual_v1.pt"

    data_loader = EEGSciencedirectRoughIterator() 
    latent_image_dataset = LatentImageSciencedirectDataset(
        gpu=validation_conf["gpu"],
        load_model=False,
        mode="latent_diff",
    )
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="training",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )
    dataset = VisualEEGEncoderTrainingDataset(
        latent_image_dataset=latent_image_dataset,
        eeg_dataset=eeg_dataset,
    )
    model = EEChannelNet(conf=model_conf["model"])
    validation = EEGEncoderValidation(
        model=model,
        dataset=dataset,
        conf=validation_conf,
        limit_to_idxs=None
    )
    validation.load_model(validation_conf["model_save_path"])
    validation.validate()