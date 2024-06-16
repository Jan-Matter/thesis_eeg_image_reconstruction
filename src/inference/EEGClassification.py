import sys
import os
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from dotenv import find_dotenv, load_dotenv

#from torch.utils.tensorboard import SummaryWriter

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.datasets.ClassificationEEGEncoderTrainingDatasets.ImageClassEEGEncoderTrainingDataset import ImageClassEEGEncoderTrainingDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset

from src.datasets.LinguisticEEGEncoderTrainingDataset import LinguisticEEGEncoderTrainingDataset


from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.eeg_encoders.models.lstm_encoder_classifier import LSTMEncoderClassifier

from eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier




class EEGClassification:

    def __init__(self, model, dataset, conf, limit_to_idxs=None):
       
        if conf["gpu"] > 0:
            torch.cuda.set_device(conf["gpu"])
        self.__device = torch.device(f'cuda:{conf["gpu"]}') if torch.cuda.is_available() else torch.device("cpu")
        self.__all_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']
        self.__channels_used = np.array([i for i, channel in enumerate(self.__all_channels) if channel in conf["limit_to_channels"]])
        self.__freq_bands_used = conf["limit_to_freq_bands"]
        
        self.__dataset = Subset(dataset, limit_to_idxs) if limit_to_idxs is not None else dataset
        self.__model = model.to(self.__device)
        idxs = limit_to_idxs if limit_to_idxs is not None else list(np.arange(len(self.__dataset)))
        
        self.__test_data = torch.stack([self.__dataset[i]['latent'].reshape(-1) for i in idxs]).double()
        self.__labels = torch.stack([self.__dataset[i]["class"] for i in idxs]).double()




    def classify(self, idx):
        self.__model.eval()
        data = self.__dataset[idx]

        data["eeg_data"] = torch.from_numpy(data["eeg_data"]).unsqueeze(0).to(self.__device)
        data["eeg_data"] = self.__filter_data(data["eeg_data"])

        with torch.no_grad():
            output = self.__model(data["eeg_data"]).double().cpu()
            #output = self.__model.linear_head(output_pre)
            similarity = (output @ self.__test_data.t()).softmax(dim=-1)

           # _, indices = similarity.topk(5)
            top1 = torch.argmax(similarity).item()
            #top5 = indices.numpy().tolist()[0]

            class_vec = data['class']

            if torch.sum(class_vec) == 0:
                true_class = -1
            else:
                true_class = torch.argmax(class_vec).item()

            top1_output_class = torch.argmax(self.__labels[top1]).item()
            #top5_output_classes = [torch.argmax(self.__labels[i]).item() for i in top5]

        return {
            "true_class": true_class,
            "output_class": top1_output_class,
            "true_img_id": data["img_id"],
            "predicted_img_id": self.__dataset[top1]["img_id"],
            "top_1_correct": top1_output_class == true_class
            #"top_5_correct": true_class in top5_output_classes,
        }
    

    def __filter_data(self, data):
        data = data[:, :, self.__channels_used, :]
        return data
        

if __name__ == "__main__":
    classification_conf_path = Path(os.getenv("CONFIG_FOLDER")) / "inference" / "eeg_classification.yaml"
    classification_conf = yaml.load(classification_conf_path.open(), Loader=yaml.FullLoader)["classification"]

    model_conf_path = Path(classification_conf['model_conf_path'])
    model_conf = yaml.load(model_conf_path.open(), Loader=yaml.FullLoader)["model"]

    data_loader = EEGSciencedirectRoughIterator() 
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="test",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )
    latent_text_dataset = LatentTextSciencedirectDataset(load_model=False)
    latent_dataset = LinguisticEEGEncoderTrainingDataset(
        eeg_dataset=eeg_dataset,
        latent_text_dataset=latent_text_dataset,
        limit_to_classes=None,
        equalize_classes=False
    )
    dataset = ImageClassEEGEncoderTrainingDataset(
        latent_dataset=latent_dataset,
        limit_to_classes=None,
        equalize_classes=False
    )

    #has to be set manually
    model_conf["modules"]["fc"]["out_features"] = 768
    model = LSTMEncoderClassifier(conf=model_conf)
    #model.load_state_dict(torch.load(classification_conf["model_save_path"]))

    latent_text_dataset = LatentTextSciencedirectDataset(load_model=False)

    test_data = torch.stack([latent_text_dataset[i].squeeze(0) for i in range(1, 101)])
    queryier = EEGScienceDirectSQLDBQueryier()
    
    
    classification = EEGClassification(
        model,
        dataset,
        mode="visual",
        conf=classification_conf
    )
    output = classification.classify(1)
    print(output)
    
