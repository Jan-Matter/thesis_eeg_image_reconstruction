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
from dotenv import find_dotenv, load_dotenv
#from torch.utils.tensorboard import SummaryWriter

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.datasets.ClassificationEEGEncoderTrainingDatasets.BaseClassificationEEGEncoderTrainingDataset import ClassificationEEGEncoderTrainingDataset
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.eeg_encoders.models.eeg_channel_net import EEChannelNet
from src.eeg_encoders.models.lstm_encoder_classifier import LSTMEncoderClassifier

from src.datasets.ClassificationEEGEncoderTrainingDatasets.ImageClassEEGEncoderTrainingDataset import ImageClassEEGEncoderTrainingDataset
from src.datasets.ClassificationEEGEncoderTrainingDatasets.ImageConceptEEGEncoderTrainingDataset import ImageConceptEEGEncoderTrainingDataset
from src.datasets.ClassificationEEGEncoderTrainingDatasets.ObjectPositionEEGEncoderTrainingDataset import ObjectPositionEEGEncoderTrainingDataset
from src.datasets.ClassificationEEGEncoderTrainingDatasets.BackgroundColorEEGEncoderTrainingDataset import BackgroundColorEEGEncoderTrainingDataset
from src.datasets.ClassificationEEGEncoderTrainingDatasets.ObjectHeightEEGEncoderTrainingDataset import ObjectHeightEEGEncoderTrainingDataset
from src.datasets.ClassificationEEGEncoderTrainingDatasets.ObjectWidthEEGEncoderTrainingDataset import ObjectWidthEEGEncoderTrainingDataset
from sklearn.model_selection import train_test_split
from src.losses.contrastive_info_nce_loss import ContrastiveInfoNCELoss


class EEGClassificationFineTuning:

    def __init__(self, model, dataset, conf, limit_to_idxs=None):
        idxs = np.arange(len(dataset)) if limit_to_idxs is None else limit_to_idxs
        self.__train_size = int(conf["train_split"] * len(idxs))
        self.__batch_size = conf["batch_size"]
        self.__epochs = conf["epochs"]
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
        
        self.__pretrained_model_save_path = conf["pretrained_model_save_path"]
        #model.load_state_dict(torch.load(self.__pretrained_model_save_path))

        self.__losses_save_path = conf["losses_save_path"]

        tensorboard_folder = os.getenv("TENSORBOARD_FOLDER") + f"/visual_eeg_encoder_classifier"
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        #self.__writer = SummaryWriter(os.getenv("TENSORBOARD_FOLDER") + f"/visual_eeg_encoder_classifier")

        if conf["gpu"] > 0:
            torch.cuda.set_device(conf["gpu"])
        self.__device = torch.device(f'cuda:{conf["gpu"]}') if torch.cuda.is_available() else torch.device("cpu")

        train_indices, validation_indices = train_test_split(idxs, train_size=self.__train_size, test_size=len(idxs)-self.__train_size, shuffle=True)

        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        self.dataset = dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.__batch_size, shuffle=True)
        self.__model = model.to(self.__device)

        if self.__model.__class__.__name__ == "LSTMEncoderClassifier":
            pass
            # Freeze the lstm_layers
            #for param in self.__model.lstm_layers.parameters():
            #    param.requires_grad = False
            #for param in self.__model.fc.parameters():
            #    param.requires_grad = False
            
        self.__optimizer = Adam(self.__model.parameters(), lr=conf["learning_rate"])
        self.__loss = ContrastiveInfoNCELoss(temperature=1)

        self.__val_losses = []
        self.__train_losses = []
        self.__best_loss = 1000
        self.__equalize_classes = conf["equalize_classes"]
    
    @property
    def best_precision(self):
        return self.__best_precision
    

    def train(self):
        self.__save_model() #save initial model
        self.validate(0) #validate initial model
        for epoch in range(1, self.__epochs + 1):
            print(f"Epoch: {epoch}")
            self.__batch_losses = []
            for batch in tqdm(self.train_loader):
                self.__optimizer.zero_grad()
                batch = self.__filter_batch(batch)
                output = self.__model(batch["eeg_data"])
                #output = self.__model.linear_head(output_pre)

                #output = self.__model.fc(output_pre)

                latent = batch["latent"].squeeze(1).to(self.__device)

                loss = self.__loss(output, latent)

                #loss = self.__loss(output_classes, img_concept)
                loss.backward()
                self.__optimizer.step()
                #self.__writer.add_scalar("Loss/train", loss.item(), epoch)
                print(f"Batch Loss: {loss.item()}")
                self.__batch_losses.append(loss.item())
            self.__train_losses.append(np.mean(np.array(self.__batch_losses)))
        
            loss = self.validate(epoch)
            if loss < self.__best_loss:
                self.__best_loss = loss
                self.__save_model()
            self.__save_losses()
        
            #stop criterion
            if epoch > 20 and np.max(self.__val_losss[-20:]) > self.__best_loss:
                print("Early Stopping")
                break
    

    def validate(self, epoch):
        with torch.no_grad():
            self.__model.eval()
            losses = []
            true_positives = [0 for i in range(self.dataset.num_classes)]
            false_positives = [0 for i in range(self.dataset.num_classes)]
            for batch in self.validation_loader:
                batch = self.__filter_batch(batch)
                output = self.__model(batch["eeg_data"])
                #output = self.__model.fc(output_pre)

                latent = batch["latent"].squeeze(1).to(self.__device)

                loss = self.__loss(output, latent)



        # self.__writer.add_scalar("precision/validation", precision, epoch)
        loss_value = loss.item()
        print(f"Loss : {loss_value}")
        self.__val_precisions.append(loss_value)
        self.__model.train()
        return loss_value
    
    def load_model(self, model_save_path):
        self.__model.load_state_dict(torch.load(model_save_path))
        self.__model.eval()

    def __save_model(self):
        torch.save(self.__model.state_dict(),
                    self.__model_save_path)
    
    def __save_losses(self):
        accuracies = np.array(self.__val_precisions)
        np.save(self.__losses_save_path, accuracies)
    
    def __filter_batch(self, batch):
        outbatch = batch.copy()
        
        outbatch["eeg_data"] = outbatch["eeg_data"][:, :, self.__channels_used, :]
        return outbatch


if __name__ == '__main__':
    training_conf_path = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_finetuning.yaml"
    training_conf = yaml.load(training_conf_path.open(), Loader=yaml.FullLoader)["training"]

    model_conf_path = Path(training_conf['model_conf_path'])
    model_conf = yaml.load(model_conf_path.open(), Loader=yaml.FullLoader)["model"]

    data_loader = EEGSciencedirectRoughIterator() 
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="training",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )
    dataset = ImageClassEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        limit_to_classes=None
    )

    #has to be set manually
    model_conf["modules"]["fc"]["out_features"] = dataset.num_classes
    model = LSTMEncoderClassifier(conf=model_conf)
    training_conf['pretrained_model_save_path'] = os.getenv("MODEL_FOLDER") + "/eeg_img_classification_pretraining_e1.pt"
    training_conf['model_save_path'] = os.getenv("MODEL_FOLDER") + "/eeg_img_classification_finetuning_e1_test.pt"
    training_conf['losses_save_path'] = os.getenv("OUTPUT_FOLDER") + "/losses/eeg_img_classification_finetuning_e1_losses_test.npy"
    
    training = EEGClassificationFineTuning(
        model,
        dataset,
        conf=training_conf
    )
    training.train()
