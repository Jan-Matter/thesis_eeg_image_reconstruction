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

from src.datasets.LinguisticEEGEncoderTrainingDataset import LinguisticEEGEncoderTrainingDataset
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.eeg_encoders.models.eeg_channel_net import EEChannelNet

class LinguisticEEGEncoderTraining:

    def __init__(self, model, dataset, conf, limit_to_idxs):
        idxs = np.arange(len(dataset)) if limit_to_idxs is None else limit_to_idxs
        self.__train_size = int(conf["train_split"] * len(idxs))
        self.__batch_size = conf["batch_size"]
        self.__epochs = conf["epochs"]
        self.__validate_every = conf["validate_every"]
        self.__model_save_path = conf["model_save_path"]
        self.__losses_save_path = conf["losses_save_path"]
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

        train_indices, validation_indices = train_test_split(idxs, train_size=self.__train_size, test_size=len(idxs)-self.__train_size, shuffle=True)

        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        self.dataset = dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=False)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.__batch_size, shuffle=False)
        self.__model = model.to(self.__device)
        self.__optimizer = Adam(self.__model.parameters(), lr=conf["learning_rate"])
        self.__loss = nn.MSELoss()
        self.__train_losses = []
        self.__best_loss = np.inf

    def train(self):
        self.__save_model() #save initial model
        self.__train_losses = []
        for epoch in range(1, self.__epochs + 1):
            self.__batch_losses = []
            self.__model.train()
            count = 0
            for batch in tqdm(self.train_loader):
                batch = self.__filter_batch(batch)
                latent_image = batch["latent"].to(self.__device)
                self.__optimizer.zero_grad()
                output = self.__model(batch["eeg_data"]).double()
                latent_image_flattened = torch.flatten(latent_image, start_dim=1).double()
                loss = self.__loss(output, latent_image_flattened)
                loss.backward()
                self.__optimizer.step()
                self.__batch_losses.append(loss.item())
                #self.__writer.add_scalar('Loss/train', loss.item(), epoch)
                print(f"Epoch Batch Loss: {loss.item()}")
                if count % self.__validate_every == 0:
                    pass
                    #mean_validation_loss = self.validate(epoch)
                    #print(f"Epoch: {epoch}, Mean Validation Loss: {mean_validation_loss}")
                count += 1

            mean_validation_loss = self.validate(epoch)
            print(f"Epoch: {epoch}, Mean Validation Loss: {mean_validation_loss}")
            self.__train_losses.append(mean_validation_loss)
            if mean_validation_loss < self.__best_loss:
                self.__best_loss = mean_validation_loss
                self.__save_model()
            self.__save_losses()
    
    def validate(self, epoch):
        with torch.no_grad():
            self.__model.eval()
            losses = []
            for batch in self.validation_loader:
                batch = self.__filter_batch(batch)
                latent_image = batch["latent"].to(self.__device)
                output = self.__model(batch["eeg_data"])
                latent_image_flattened = torch.flatten(latent_image, start_dim=1)
                loss = self.__loss(output, latent_image_flattened)
                self.__batch_losses.append(loss.item())
            losses.append(np.mean(np.array(self.__batch_losses)))
        
            mean_train_loss = np.mean(losses)
            #self.__writer.add_scalar('Loss/validation', mean_train_loss, epoch)
        return mean_train_loss

    
    def load_model(self, model_save_path):
        self.__model.load_state_dict(torch.load(model_save_path))
        self.__model.eval()

    def __save_model(self):
        torch.save(self.__model.state_dict(),
                    self.__model_save_path)
    
    def __save_losses(self):
        losses = np.array(self.__train_losses)
        np.save(self.__losses_save_path, losses)
    

    def __filter_batch(self, batch):
        outbatch = batch.copy()
        
        outbatch["eeg_data"] = outbatch["eeg_data"][:, :, self.__channels_used, :]
        return outbatch


if __name__ == '__main__':
    training_conf_path = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_to_latent_text_training.yaml"
    training_conf = yaml.load(training_conf_path.open(), Loader=yaml.FullLoader)['training']
    model_conf_path = Path(os.getenv("CONFIG_FOLDER"))/  "model" / "text" / f"eeg_channel_net.yaml"
    model_conf = yaml.load(model_conf_path.open(), Loader=yaml.FullLoader)

    training_conf["model_save_path"] = os.getenv("MODEL_FOLDER") + f"/eeg_channel_net_linguistic_v1.pt"
    training_conf["losses_save_path"] = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_channel_net_linguistic_v1.npy"

    data_loader = EEGSciencedirectRoughIterator() 
    latent_text_dataset = LatentTextSciencedirectDataset(
        gpu=training_conf["gpu"],
        load_model=False
    )
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="training",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )
    dataset = LinguisticEEGEncoderTrainingDataset(
        latent_text_dataset=latent_text_dataset,
        eeg_dataset=eeg_dataset,
    )
    model = EEChannelNet(conf=model_conf["model"])
    training = LinguisticEEGEncoderTraining(
        model,
        dataset,
        conf=training_conf,
        limit_to_idxs=None
    )
    #training.load_model(training_conf["model_save_path"])
    training.train()