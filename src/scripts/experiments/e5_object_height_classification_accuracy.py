import sys
import os
import yaml
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json
import torch
from sklearn.model_selection import train_test_split
#from torch.utils.tensorboard import SummaryWriter

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.datasets.ClassificationEEGEncoderTrainingDatasets.ObjectHeightEEGEncoderTrainingDataset import ObjectHeightEEGEncoderTrainingDataset
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.datasets.LinguisticEEGEncoderTrainingDataset import LinguisticEEGEncoderTrainingDataset

from src.eeg_encoders.models.lstm_encoder_classifier import LSTMEncoderClassifier
from src.eeg_encoders.models.visual_eegnetv4_classifier import VisualEEGNetV4Classifier
from src.eeg_encoders.models.eeg_channel_net import EEChannelNet
from src.eeg_encoders.models.attention_encoder_classifier import AttentionEncoderClassifier
from src.eeg_encoders.models.eeg_gnn_classifier import GNNEEGClassifier
from src.eeg_encoders.models.benchmark_classifier import BenchmarkClassifier

from src.training.EEGSelfsupervisedContrastivePretraining import EEGSelfsupervisedContrastivePretraining
from src.training.VisualEEGFinetuning import VisualEEGEncoderTraining
from src.training.LinguisticEEGFinetuning import LinguisticEEGEncoderTraining
from src.inference.EEGClassification import EEGClassification
from torch.utils.data import Subset, Dataset

MODEL = "EEGChannelNet"
LATENT = "linguistic"
FINETUNING = True

if LATENT == "linguistic":
    LATENT_SIZE = 768
elif LATENT == "visual":
    LATENT_SIZE = 1024

if MODEL == "EEGNETV4":
    MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_eegnetv4_classification.yaml"
elif MODEL == "LSTM":
    MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_lstm_classification.yaml"
elif MODEL == "EEGChannelNet":
    MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_channelnet_classification.yaml"
elif MODEL == "EEGChannelNetGraphAttention":
    MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"
elif MODEL == "EEGChannelNetSelfAttention":
    MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_attention_classification.yaml"

PRETRAINING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_pretraining.yaml"
PRETRAINED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_pretraining_e5_{MODEL}.pt"
PRETRAINED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_pretraining_e5_{MODEL}_losses.npy"
FINETUNING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_finetuning.yaml"
FINETUNED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_finetuning_e5_{MODEL}.pt"
FINETUNED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_finetuning_e5_{MODEL}_losses.npy"
CLASSIFICATION_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "inference" / "eeg_classification.yaml"
RESULT_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/experiment_results/eeg_classification_e5_{MODEL}_results_v1.json"
SUBJECTS = [1]
RUN_TRAINING = True
EPOCHS = 100
TRAIN_SIZE = 0.8
LIMIT_TO_CLASSES = None

HEIGHTS = [
    'small',
    'medium',
    'large'
]


def prepare_training_dataset():
     
    data_loader = EEGSciencedirectRoughIterator() 
    latent_text_dataset = LatentTextSciencedirectDataset(
        gpu=0,
        load_model=True
    )
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=SUBJECTS,
            limit_to_split="training",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )

    latent_dataset = LinguisticEEGEncoderTrainingDataset(
        latent_text_dataset=latent_text_dataset,
        eeg_dataset=eeg_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        equalize_classes=False
    )

    classification_dataset = ObjectHeightEEGEncoderTrainingDataset(
        latent_dataset=latent_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        equalize_classes=True
    )
    return classification_dataset


def prepare_test_dataset():

    data_loader = EEGSciencedirectRoughIterator() 
    latent_text_dataset = LatentTextSciencedirectDataset(
        gpu=0,
        load_model=True
    )
    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=SUBJECTS,
            limit_to_split="test",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )

    latent_dataset = LinguisticEEGEncoderTrainingDataset(
        latent_text_dataset=latent_text_dataset,
        eeg_dataset=eeg_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        equalize_classes=False
    )

    classification_dataset = ObjectHeightEEGEncoderTrainingDataset(
        latent_dataset=latent_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        equalize_classes=False
    )
    return classification_dataset


def run_training(train_dataset, limit_to_idxs=None):
    #pretraining of lstm classifier

    model_conf_path = MODEL_CONF_PATH
    model_conf = yaml.load(model_conf_path.open(), Loader=yaml.FullLoader)["model"]

    training_conf_path = PRETRAINING_CONF_PATH
    training_conf = yaml.load(training_conf_path.open(), Loader=yaml.FullLoader)["training"]

    #set training conf
    training_conf["model_save_path"] = PRETRAINED_MODEL_SAVE_PATH
    training_conf["losses_save_path"] = PRETRAINED_LOSS_SAVE_PATH
    training_conf["equalize_classes"] = False
    training_conf["epochs"] = EPOCHS

    if MODEL == "EEGNETV4":
        model_conf["n_outputs"] = LATENT_SIZE
        model = VisualEEGNetV4Classifier(conf=model_conf)
    elif MODEL == "LSTM":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = LSTMEncoderClassifier(conf=model_conf)
    elif MODEL == "EEGChannelNet":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = EEChannelNet(conf=model_conf)
    elif MODEL == "EEGChannelNetGraphAttention":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = GNNEEGClassifier(conf=model_conf)
    elif MODEL == "EEGChannelNetSelfAttention":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = AttentionEncoderClassifier(conf=model_conf)
    elif MODEL == "Benchmark":
        model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE)
    
    training = EEGSelfsupervisedContrastivePretraining(
        model,
        train_dataset,
        conf=training_conf,
        limit_to_idxs=limit_to_idxs
    )
    training.train()

    if FINETUNING:
    #finetuning of lstm classifier

        training_conf_path = FINETUNING_CONF_PATH
        training_conf = yaml.load(training_conf_path.open(), Loader=yaml.FullLoader)["training"]

        #set training conf
        training_conf["model_save_path"] = FINETUNED_MODEL_SAVE_PATH
        training_conf["losses_save_path"] = FINETUNED_LOSS_SAVE_PATH
        training_conf["pretrained_model_save_path"] = PRETRAINED_MODEL_SAVE_PATH
        training_conf["epochs"] = EPOCHS

        params = torch.load(training_conf["pretrained_model_save_path"])
        if MODEL == "EEGNETV4":
            model_conf["n_outputs"] = LATENT_SIZE
            params = {k: v for k, v in params.items() if not 'final_layer' in k}
            model = VisualEEGNetV4Classifier(conf=model_conf)
            #model.load_state_dict(params, strict=False)
        elif MODEL == "LSTM":
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            params = {k: v for k, v in params.items() if not 'linear_head' in k and not 'fc' in k}
            model = LSTMEncoderClassifier(conf=model_conf)
            model.load_state_dict(params, strict=False)
        elif MODEL == "EEGChannelNet":
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            params = {k: v for k, v in params.items() if not'fc' in k}
            model = EEChannelNet(conf=model_conf)
            model.load_state_dict(params, strict=False)
        elif MODEL == "EEGChannelNetGraphAttention":
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            params = {k: v for k, v in params.items() if not 'fc' in k}
            model = GNNEEGClassifier(conf=model_conf)
            model.load_state_dict(params, strict=False)
        elif MODEL == "EEGChannelNetSelfAttention":
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            params = {k: v for k, v in params.items() if not 'fc' in k}
            model = AttentionEncoderClassifier(conf=model_conf)
            model.load_state_dict(params, strict=False)
        elif MODEL == "Benchmark":
            model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE)
            model.load_state_dict(params, strict=False)
        
        if LATENT == "linguistic":
            training = LinguisticEEGEncoderTraining(
                model,
                train_dataset,
                conf=training_conf,
                limit_to_idxs=limit_to_idxs
            )
        elif LATENT == "visual":
            training = VisualEEGEncoderTraining(
                model,
                train_dataset,
                conf=training_conf,
                limit_to_idxs=limit_to_idxs
            )
        training.train()


def evaluate(test_dataset, limit_to_idxs=None):
    classification_conf_path = CLASSIFICATION_CONF_PATH
    classification_conf = yaml.load(classification_conf_path.open(), Loader=yaml.FullLoader)["classification"]

    classification_conf["model_save_path"] = FINETUNED_MODEL_SAVE_PATH
    classification_conf["model_conf_path"] = MODEL_CONF_PATH

    model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]

    #has to be set manually
    if MODEL == "EEGNETV4":
        model_conf["n_outputs"] = LATENT_SIZE
        model = VisualEEGNetV4Classifier(conf=model_conf)
    elif MODEL == "LSTM":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = LSTMEncoderClassifier(conf=model_conf)
    elif MODEL == "EEGChannelNet":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = EEChannelNet(conf=model_conf)
    elif MODEL == "EEGChannelNetGraphAttention":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = GNNEEGClassifier(conf=model_conf)
    elif MODEL == "EEGChannelNetSelfAttention":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model = AttentionEncoderClassifier(conf=model_conf)
    elif MODEL == "Benchmark":
        model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE)

    model.load_state_dict(torch.load(classification_conf["model_save_path"]))
    
    classification = EEGClassification(
        model,
        test_dataset,
        conf=classification_conf,
        limit_to_idxs=limit_to_idxs
    )


    height_statistics = [
        {
        "height": c,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
        }
        for c in HEIGHTS
    ]
    
    
    length = len(test_dataset) if limit_to_idxs is None else len(limit_to_idxs)
    for i in range(length):
        output = classification.classify(i)
        true_class = output["true_class"]
        if true_class == -1:
            continue
        output_class = output["output_class"]
        if true_class == output_class:
            height_statistics[true_class]["true_positives"] += 1
            for c in range(len(HEIGHTS)):
                if c != true_class:
                    height_statistics[c]["true_negatives"] += 1
        else:
            height_statistics[true_class]["false_negatives"] += 1
            height_statistics[output_class]["false_positives"] += 1
            for c in range(len(HEIGHTS)):
                if c != true_class and c != output_class:
                    height_statistics[c]["true_negatives"] += 1

    for c in range(len(HEIGHTS)):
        true_positives = height_statistics[c]["true_positives"]
        false_positives = height_statistics[c]["false_positives"]
        true_negatives = height_statistics[c]["true_negatives"]
        false_negatives = height_statistics[c]["false_negatives"]
        height_statistics[c]["accuracy"] = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
        height_statistics[c]["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        height_statistics[c]["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        height_statistics[c]["f1"] = 2 * (height_statistics[c]["precision"] * height_statistics[c]["recall"]) / (height_statistics[c]["precision"] + height_statistics[c]["recall"]) if (height_statistics[c]["precision"] + height_statistics[c]["recall"]) > 0 else 0
    
    mean_accuracy = sum([c["accuracy"] for c in height_statistics]) / len(height_statistics)
    mean_precision = sum([c["precision"] for c in height_statistics]) / len(height_statistics)
    mean_recall = sum([c["recall"] for c in height_statistics]) / len(height_statistics)
    mean_f1 = sum([c["f1"] for c in height_statistics]) / len(height_statistics)
    print("Mean accuracy: ", mean_accuracy)
    print("Mean precision: ", mean_precision)
    print("Mean recall: ", mean_recall)
    print("Mean f1: ", mean_f1)

    return {
        "mean_accuracy": mean_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "color_statistics": height_statistics,
    }



def main(training=True):
    training_dataset = prepare_training_dataset()
    test_dataset = prepare_test_dataset()
    #train_idxs, test_idxs = train_test_split(range(len(dataset)), train_size=TRAIN_SIZE)
    if training:
        run_training(train_dataset=training_dataset, limit_to_idxs=None)
    class_statistics = evaluate(test_dataset=test_dataset, limit_to_idxs=None)
    with(open(RESULT_SAVE_PATH, "w")) as f:
        json.dump(class_statistics, f, indent=4)

    
    
    
if __name__ == "__main__":
    
    for model in ["Benchmark", "EEGNETV4", "EEGChannelNetSelfAttention", "EEGChannelNet", "EEGChannelNetGraphAttention", "LSTM"]:
        MODEL = model

        if MODEL == "EEGNETV4":
            MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_eegnetv4_classification.yaml"
        elif MODEL == "LSTM":
            MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_lstm_classification.yaml"
        elif MODEL == "EEGChannelNet":
            MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_channelnet_classification.yaml"
        elif MODEL == "EEGChannelNetGraphAttention":
            MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"
        elif MODEL == "EEGChannelNetSelfAttention":
            MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_attention_classification.yaml"

        subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        for subject in subjects:
            PRETRAINING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_pretraining.yaml"
            PRETRAINED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e5_no_finetuning_s{subject}_{MODEL}.pt"
            PRETRAINED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_pretraining_e5_no_finetuning_s{subject}_{MODEL}_losses.npy"
            FINETUNING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_finetuning.yaml"
            FINETUNED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e5_no_finetuning_s{subject}_{MODEL}.pt"
            FINETUNED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_finetuning_e5_no_finetuning_s{subject}_{MODEL}_losses.npy"
            CLASSIFICATION_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "inference" / "eeg_classification.yaml"
            RESULT_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/experiment_results/finetuning_s{subject}/eeg_classification_e5_no_finetuning_s{subject}_{MODEL}_results_v1.json"

            os.makedirs(os.getenv("OUTPUT_FOLDER") + f"/experiment_results/finetuning_s{subject}", exist_ok=True)
            FINETUNING = False

            SUBJECTS = [subject]
            
            main(training=RUN_TRAINING)