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

from src.datasets.ClassificationEEGEncoderTrainingDatasets.ImageConceptEEGEncoderTrainingDataset import ImageConceptEEGEncoderTrainingDataset
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
import random



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
PRETRAINED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e9_{MODEL}.pt"
PRETRAINED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_pretraining_e9_{MODEL}_losses.npy"
FINETUNING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_finetuning.yaml"
FINETUNED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e9_{MODEL}.pt"
FINETUNED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_finetuning_e9_{MODEL}_losses.npy"
CLASSIFICATION_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "inference" / "eeg_classification.yaml"
RESULT_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/experiment_results/eeg_classification_e9_{MODEL}_results_v1.json"
LIMIT_TO_CLASSES = None
EPOCHS = 1
TRAIN_SIZE = 0.8
SEED = 42
random.seed(SEED)
SUBJECTS = [1]

RUN_TRAINING = True

def prepare_training_dataset(limit_to_concepts=None):
     
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

    classification_dataset = ImageConceptEEGEncoderTrainingDataset(
        latent_dataset=latent_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        limit_to_concepts=limit_to_concepts,
        equalize_classes=True
    )
    return classification_dataset


def prepare_test_dataset(limit_to_concepts=None):

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

    classification_dataset = ImageConceptEEGEncoderTrainingDataset(
        latent_dataset=latent_dataset,
        limit_to_classes=LIMIT_TO_CLASSES,
        limit_to_concepts=limit_to_concepts,
        equalize_classes=False
    )
    return classification_dataset


def run_training(train_dataset, limit_to_idxs=None, limit_to_channels=None):
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
    if limit_to_channels is not None:
        training_conf["limit_to_channels"] = limit_to_channels

    if MODEL == "EEGNETV4":
        model_conf["n_outputs"] = LATENT_SIZE
        model_conf["n_chans"] = len(limit_to_channels)
        model = VisualEEGNetV4Classifier(conf=model_conf)
    elif MODEL == "LSTM":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model_conf["modules"]["lstm"]["in_channels"] = len(limit_to_channels)
        model_conf["modules"]["lstm"]["out_channels"] = len(limit_to_channels)
        model_conf["modules"]["fc"]["in_features"] = len(limit_to_channels)
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
        if limit_to_channels is not None:
            training_conf["limit_to_channels"] = limit_to_channels

        params = torch.load(training_conf["pretrained_model_save_path"])
        if MODEL == "EEGNETV4":
            model_conf["n_outputs"] = LATENT_SIZE
            model_conf["n_chans"] = len(limit_to_channels)
            params = {k: v for k, v in params.items() if not 'final_layer' in k}
            model = VisualEEGNetV4Classifier(conf=model_conf)
            #model.load_state_dict(params, strict=False)
        elif MODEL == "LSTM":
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
            model_conf["modules"]["lstm"]["in_channels"] = len(limit_to_channels)
            model_conf["modules"]["lstm"]["out_channels"] = len(limit_to_channels)
            model_conf["modules"]["fc"]["in_features"] = len(limit_to_channels)
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


def evaluate(test_dataset, limit_to_idxs: [list, None]=None, limit_to_channels=None):
    classification_conf_path = CLASSIFICATION_CONF_PATH
    classification_conf = yaml.load(classification_conf_path.open(), Loader=yaml.FullLoader)["classification"]

    classification_conf["model_save_path"] = FINETUNED_MODEL_SAVE_PATH
    classification_conf["model_conf_path"] = MODEL_CONF_PATH
    if limit_to_channels is not None:
        classification_conf["limit_to_channels"] = limit_to_channels

    model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]

    #has to be set manually
    if MODEL == "EEGNETV4":
        model_conf["n_outputs"] = LATENT_SIZE
        model_conf["n_chans"] = len(limit_to_channels)
        model = VisualEEGNetV4Classifier(conf=model_conf)
    elif MODEL == "LSTM":
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model_conf["modules"]["fc"]["out_features"] = LATENT_SIZE
        model_conf["modules"]["lstm"]["in_channels"] = len(limit_to_channels)
        model_conf["modules"]["lstm"]["out_channels"] = len(limit_to_channels)
        model_conf["modules"]["fc"]["in_features"] = len(limit_to_channels)
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

    concept_ids = test_dataset.concept_ids
    concept_statistics = [
        {
        "concept": c,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
        }
        for c in concept_ids
    ]
    
    length = len(test_dataset) if limit_to_idxs is None else len(limit_to_idxs)
    
    for i in range(length):
        output = classification.classify(i)
        true_class = output["true_class"]
        output_class = output["output_class"]
        if true_class == output_class:
            concept_statistics[true_class]["true_positives"] += 1
            for c in range(len(concept_ids)):
                if c != true_class:
                    concept_statistics[c]["true_negatives"] += 1
        else:
            concept_statistics[true_class]["false_negatives"] += 1
            concept_statistics[output_class]["false_positives"] += 1
            for c in range(len(concept_ids)):
                if c != true_class and c != output_class:
                    concept_statistics[c]["true_negatives"] += 1

    for c in range(len(concept_ids)):
        true_positives = concept_statistics[c]["true_positives"]
        false_positives = concept_statistics[c]["false_positives"]
        true_negatives = concept_statistics[c]["true_negatives"]
        false_negatives = concept_statistics[c]["false_negatives"]
        concept_statistics[c]["accuracy"] = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
        concept_statistics[c]["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        concept_statistics[c]["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        concept_statistics[c]["f1"] = 2 * (concept_statistics[c]["precision"] * concept_statistics[c]["recall"]) / (concept_statistics[c]["precision"] + concept_statistics[c]["recall"]) if (concept_statistics[c]["precision"] + concept_statistics[c]["recall"]) > 0 else 0
    
    mean_accuracy = sum([c["accuracy"] for c in concept_statistics]) / len(concept_statistics)
    mean_precision = sum([c["precision"] for c in concept_statistics]) / len(concept_statistics)
    mean_recall = sum([c["recall"] for c in concept_statistics]) / len(concept_statistics)
    mean_f1 = sum([c["f1"] for c in concept_statistics]) / len(concept_statistics)
    print("Mean accuracy: ", mean_accuracy)
    print("Mean precision: ", mean_precision)
    print("Mean recall: ", mean_recall)
    print("Mean f1: ", mean_f1)

    return {
        "mean_accuracy": mean_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "concept_statistics": concept_statistics,
    }


def main(training=True):

    eeg_groups = {
        "frontal": ["Fp1", "Fp2", "AF3", "AFz", "AF4", "F1", "F2"],
        "central": [ "FC1", "FCz", "FC2", "C3", "C1", "Cz", "C2", "C4"],
        "parietal": ["CP3", "CP1", "CPz", "CP2", "CP4", "P5", "P3", "P1", "Pz", "P2", "P4", "P6"],
        "occipital": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
        "left_temporal": ["AF7", "F7", "F5", "F3", "FT7", "FC5", "FC3", "C5", "TP7", "CP5", "P7"],
        "right_temporal": ["AF8", "F4", "F6", "F8", "FC4", "FC6", "FT8", "C6", "CP6", "TP8", "P8"],
        "all": ["Fp1", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8"]
    }

    #(concept_count, repetitions)
    training_cases = [
        (eeg_groups["frontal"], "frontal"),
        (eeg_groups["central"], "central"),
        (eeg_groups["parietal"], "parietal"),
        (eeg_groups["occipital"], "occipital"),
        (eeg_groups["left_temporal"], "left_temporal"),
        (eeg_groups["right_temporal"], "right_temporal"),
        (eeg_groups["all"], "all"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["frontal"])), "all_but_frontal"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["central"])), "all_but_central"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["parietal"])), "all_but_parietal"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["occipital"])), "all_but_occipital"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["left_temporal"])), "all_but_left_temporal"),
        (list(set(eeg_groups["all"]) - set(eeg_groups["right_temporal"])), "all_but_right_temporal"),
    ]

    results = []

    run_pretraining = True
    for i, (limit_to_channels, name) in enumerate(training_cases):
        print("Training case ", i + 1)

        train_dataset = prepare_training_dataset(None)
        test_dataset = prepare_test_dataset(None)
        #train_idxs, test_idxs =  train_test_split(range(len(dataset)), train_size=TRAIN_SIZE)
        if training:
            run_training(train_dataset, None, limit_to_channels)
        channel_result = evaluate(test_dataset, None, limit_to_channels)

        result = {
            "name": name,
            "channels": limit_to_channels,
            "result": channel_result
        }
        results.append(result)
    
    output_results = {
        "results": results,
    }
    
    with open(RESULT_SAVE_PATH, "w") as f:
        json.dump(output_results, f, indent=4)



if __name__ == "__main__":
    for model in ["EEGNETV4"]:
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
            PRETRAINED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e8_no_finetuning_s{subject}_{MODEL}.pt"
            PRETRAINED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_pretraining_e8_no_finetuning_s{subject}_{MODEL}_losses.npy"
            FINETUNING_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "training" / "eeg_img_classification_finetuning.yaml"
            FINETUNED_MODEL_SAVE_PATH = os.getenv("MODEL_FOLDER") + f"/eeg_img_classification_training_e8_no_finetuning_s{subject}_{MODEL}.pt"
            FINETUNED_LOSS_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/losses/eeg_img_classification_finetuning_e8_no_finetuning_s{subject}_{MODEL}_losses.npy"
            CLASSIFICATION_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "inference" / "eeg_classification.yaml"
            RESULT_SAVE_PATH = os.getenv("OUTPUT_FOLDER") + f"/experiment_results/finetuning_s{subject}/eeg_classification_e8_no_finetuning_s{subject}_{MODEL}_results_v1.json"

            os.makedirs(os.getenv("OUTPUT_FOLDER") + f"/experiment_results/finetuning_s{subject}", exist_ok=True)
            FINETUNING = False

            SUBJECTS = [subject]
            
            main(training=RUN_TRAINING)
