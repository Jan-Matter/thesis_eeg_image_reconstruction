from PIL import Image
import os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
import sys
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.VisualEEGEncoderTrainingDataset import VisualEEGEncoderTrainingDataset
from src.datasets.LinguisticEEGEncoderTrainingDataset import LinguisticEEGEncoderTrainingDataset
from src.eeg_encoders.models.eeg_gnn_classifier import GNNEEGClassifier
from src.eeg_encoders.models.benchmark_classifier import BenchmarkClassifier
from src.inference.EEGToImageDecoder import EEGToImageDecoder




def generate_image_matrix(decoder, dataset_visual, dataset_linguistic, i, images_folder, image_class):

    IDX = i
    visual_data = dataset_visual[IDX]
    linguistic_data = dataset_linguistic[IDX]


    queryier = EEGScienceDirectSQLDBQueryier()
    folder = images_folder / f"img_mat_{image_class}_{i}{'_vis' if decoder.config['guide_visual'] else ''}{'_ling' if decoder.config['guide_linguistic'] else ''}"
    os.makedirs(folder, exist_ok=True)
    orig_img = next(queryier.run_query(f"""SELECT 
        image.img_path
        FROM image 
        WHERE image.img_id = '{linguistic_data['img_id']}'
        """))['img']
    orig_img = Image.fromarray(orig_img.astype(np.uint8)).resize((512,512))
    orig_img.save(folder /"orig_img.png")

    image_none = decoder.decode(IDX, decode_visual_eeg=False, decode_linguistic_eeg=False)
    image_all = decoder.decode(IDX, decode_visual_eeg=True, decode_linguistic_eeg=True)
    image_ling = decoder.decode(IDX, decode_visual_eeg=False, decode_linguistic_eeg=True)
    image_vis = decoder.decode(IDX, decode_visual_eeg=True, decode_linguistic_eeg=False)
    image_none.save(folder / "image_none.png")
    image_ling.save(folder / "image_ling.png")
    image_vis.save(folder / "image_vis.png")
    image_all.save(folder / "image_all.png")



def main():

    configs = OmegaConf.load("/home/matterj/codebases/eeg_image_reconstruction/src/configs/inference/eeg_to_image_decoder.yaml")
    visual_conf_path = MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"
    linguistic_conf_path = MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"

    visual_model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]

    data_loader = EEGSciencedirectRoughIterator() 

    latent_image_dataset = LatentImageSciencedirectDataset(
        gpu=0,
        load_model=True,
        mode="latent_diff"
    )

    latent_text_dataset = LatentTextSciencedirectDataset(
        gpu=0,
        load_model=True
    )


    eeg_dataset = EEGSciencedirectPreprocessedDataset(rough_data_loader=data_loader,
            limit_to_subj=[1],
            limit_to_split="test",
            avg_repetitions=True,
            in_memory_cache_behaviour="last",
            overwrite_cache=False,
            preprocess=False
        )


    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for subject in subjects:
        IMAGES_FOLDER = Path(os.getenv("OUTPUT_FOLDER") + f"/decoded/e12_{subject}")

        for img_class in ['animal', 'vehicle', 'clothing']:
            LATENT_SIZE = 1024
            visual_model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE).to('cuda')
            visual_model.load_state_dict(torch.load(Path(os.getenv("MODEL_FOLDER")) / f"eeg_img_classification_training_e10_no_finetuning_s{subject}_Benchmark.pt"))

            linguistic_model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]
            LATENT_SIZE = 768
            linguistic_model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE).to('cuda')
            linguistic_model.load_state_dict(torch.load(Path(os.getenv("MODEL_FOLDER")) / f"eeg_img_classification_training_e11_no_finetuning_s{subject}_Benchmark.pt"))

            modes = [
                {'visual': False, 'linguistic': False},
                {'visual': False, 'linguistic': True},
                {'visual': True, 'linguistic': False},
                {'visual': True, 'linguistic': True}
            ]

            for mode in modes:
            
                dataset_visual = VisualEEGEncoderTrainingDataset(
                latent_image_dataset=latent_image_dataset,
                eeg_dataset=eeg_dataset,
                limit_to_classes=[img_class]
                )

                dataset_linguistic = LinguisticEEGEncoderTrainingDataset(
                    latent_text_dataset=latent_text_dataset,
                    eeg_dataset=eeg_dataset,
                    limit_to_classes=[img_class]
                )

                configs["inference"]["guide_visual"] = mode['visual']
                configs["inference"]["guide_linguistic"] = mode['linguistic']

                decoder = EEGToImageDecoder(
                    config=configs,
                    visual_decoder=visual_model,
                    lingustic_decoder=linguistic_model,
                    linguistic_dataset=dataset_linguistic,
                    visual_dataset=dataset_visual
                )

                for i in range(len(dataset_visual)):
                    generate_image_matrix(decoder, dataset_visual, dataset_linguistic, i, IMAGES_FOLDER, img_class)


if __name__ == "__main__":
    main()


        