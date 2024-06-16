from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
from pathlib import Path
sys.path.append("../utils/")

import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.stablediffusion.models.diffusion.ddim import DDIMSampler
from src.stablediffusion.util import instantiate_from_config

from src.eeg_science_direct_sql_db.querying import EEGScienceDirectSQLDBQueryier
from src.datasets.EEGSciencedirectPreprocessedDataset import EEGSciencedirectPreprocessedDataset
from src.datasets.EEGSciencedirectRoughIterator import EEGSciencedirectRoughIterator
from src.datasets.LatentImageSciencedirectDataset import LatentImageSciencedirectDataset
from src.datasets.LatentTextSciencedirectDataset import LatentTextSciencedirectDataset
from src.datasets.VisualEEGEncoderTrainingDataset import VisualEEGEncoderTrainingDataset
from src.datasets.LinguisticEEGEncoderTrainingDataset import LinguisticEEGEncoderTrainingDataset
from src.eeg_encoders.models.eeg_gnn_classifier import GNNEEGClassifier
from src.eeg_encoders.models.benchmark_classifier import BenchmarkClassifier





class EEGToImageDecoder:

    def __init__(self, config, visual_decoder, lingustic_decoder, linguistic_dataset=None, visual_dataset=None):
        self.config = config if not 'inference' in config else config['inference']
        self.__visual_decoder  = visual_decoder
        self.__lingustic_decoder = lingustic_decoder
        self.__n_samples = 1
        self.__ddim_steps = 50
        self.__ddim_eta = 0.0
        self.__strength = 0.8
        self.__scale = 5.0
        self.__n_iter = 1
        self.__t_enc = int(self.__strength * self.__ddim_steps)
        self.__gpu = self.config['gpu']
        self.__device = torch.device(f"cuda:{self.__gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self.linguistic_dataset = linguistic_dataset
        self.visual_dataset = visual_dataset
        
        self.__queryier = EEGScienceDirectSQLDBQueryier()
        seed_everything(self.config['seed'])
        self.__diffusion_model = self.__load_model_from_config(self.config['model'], self.config['ckpt'], self.__gpu)
        device = torch.device(f"cuda:{self.__gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self.__diffusion_model = self.__diffusion_model.to(device)
        self.__diffusion_sampler = DDIMSampler(self.__diffusion_model)
        self.__diffusion_sampler.make_schedule(ddim_num_steps=self.__ddim_steps, ddim_eta=self.__ddim_eta, verbose=False)
        precision = 'autocast'
        self.__uc = self.__diffusion_model.get_learned_conditioning([""])
        self.__precision_scope = autocast if precision == "autocast" else nullcontext
        self.__guide_linguistic = self.config['guide_linguistic']
        if self.__guide_linguistic:
            idxs = list(np.arange(len(self.linguistic_dataset)))
            self.__linguistic_test_data = torch.stack([self.linguistic_dataset[i]['latent'].reshape(-1) for i in idxs]).double()
            
        self.__guide_visual = self.config['guide_visual']
        if self.__guide_visual:
            idxs = list(np.arange(len(self.visual_dataset)))
            self.__visual_test_data = torch.stack([self.visual_dataset[i]['latent'].reshape(-1) for i in idxs]).double()
            


    
    def decode(self, idx, decode_visual_eeg=True, decode_linguistic_eeg=True):
        if self.visual_dataset[idx]['img_id'] != self.linguistic_dataset[idx]['img_id']:
            raise ValueError("The visual and linguistic dataset must have the same image id")
        eeg_data = self.__prepare_eeg_data(self.visual_dataset[idx])
        if decode_visual_eeg:
            visual_latent = self.__visual_decoder(eeg_data["eeg_data"])
            if self.__guide_visual:
                #guide visual
                similarity = (visual_latent.cpu().double() @ self.__visual_test_data.t()).softmax(dim=-1)

                top1 = torch.argmax(similarity).item()
                visual_latent = self.visual_dataset[top1]['latent'].to(self.__device).unsqueeze(0)

        else:
            visual_latent = self.visual_dataset[idx]['latent'].to(self.__device)
        
        z_visual = self.__visual_first_stage_encoding(visual_latent)
        if decode_linguistic_eeg:
            linguistic_input = self.__lingustic_decoder(eeg_data["eeg_data"])
            if self.__guide_linguistic:
                #guide linguistic
                similarity = (linguistic_input.cpu().double() @ self.__linguistic_test_data.t()).softmax(dim=-1)

                top1 = torch.argmax(similarity).item()
                img_id = self.linguistic_dataset[top1]['img_id']
                concept = next(self.__queryier.run_query(f"""SELECT image.img_concept FROM image WHERE image.img_id = {img_id}"""))['img_concept']
                z_linguistic =  self.__diffusion_model.get_learned_conditioning(["A photo of a " + concept])
            else:
                z_linguistic = self.__linguistic_first_stage_encoding(linguistic_input)
        else:
            img_id = self.linguistic_dataset[idx]['img_id']
            concept = next(self.__queryier.run_query(f"""SELECT image.img_concept FROM image WHERE image.img_id = {img_id}"""))['img_concept']
            z_linguistic = self.__diffusion_model.get_learned_conditioning(["A photo of a " + concept])
        
        image = self.__generate_image(z_visual, z_linguistic)
        return image

    

    def __visual_first_stage_encoding(self, visual_latent):
        z_donwsampled = visual_latent.reshape(1, 4, 16, 16)
        z_donwsampled = z_donwsampled.to(self.__device).float()
        dec_img_downsamples = self.__diffusion_model.decode_first_stage(z_donwsampled)
        #upsample image from (3, 128, 128) to (3, 512, 512)
        dec_img = F.interpolate(dec_img_downsamples, size=(512, 512), mode='bilinear', align_corners=False)
        z = self.__diffusion_model.get_first_stage_encoding(self.__diffusion_model.encode_first_stage(dec_img))
        return z
    

    def __linguistic_first_stage_encoding(self, linguistic_input):
        c = self.__get_linguistic_latent(linguistic_input)
        #c = linguistic_input.reshape(1, 77, 768)
        return c
    
    
    def __get_linguistic_latent(self, linguistic_input):
        c = self.__uc
        c[:, 0, :] = linguistic_input
        return c
    

    def __generate_image(self, z_visual, z_linguistic):
        # Generate image from Z (image) + C (semantics)
        base_count = 0
        with torch.no_grad():
            with self.__diffusion_model.ema_scope():
                with self.__precision_scope("cuda"):
                    for n in trange(self.__n_iter, desc="Sampling"):
                        # encode (scaled latent)
                        z_visual_enc = self.__diffusion_sampler.stochastic_encode(z_visual, torch.tensor([self.__t_enc]).to(self.__device))
                        # decode it
                        samples = self.__diffusion_sampler.decode(z_visual_enc, z_linguistic, self.__t_enc, unconditional_guidance_scale=self.__scale,
                                                unconditional_conditioning=self.__uc,)

                        x_samples = self.__diffusion_model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
           
        return Image.fromarray(x_sample.astype(np.uint8))


    
    def __load_model_from_config(self, config, ckpt, gpu, verbose=False):
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
        model.cuda(f"cuda:{gpu}")
        model.eval()
        return model
    

    def __prepare_eeg_data(self, eeg_data):
        eeg_data["eeg_data"] = torch.tensor(eeg_data["eeg_data"]).unsqueeze(0).to(self.__device)
        return eeg_data
    


if __name__ == '__main__':
    configs = OmegaConf.load("/home/matterj/codebases/eeg_image_reconstruction/src/configs/inference/eeg_to_image_decoder.yaml")
    visual_conf_path = MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"
    linguistic_conf_path = MODEL_CONF_PATH = Path(os.getenv("CONFIG_FOLDER")) / "model" / "classification" / "eeg_gnn_classification.yaml"

    visual_model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]
    LATENT_SIZE = 1024
    visual_model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE).to('cuda')
    visual_model.load_state_dict(torch.load("/home/matterj/codebases/eeg_image_reconstruction/src/trained_models/eeg_img_classification_finetuning_e10_Benchmark_v4.pt"))
    linguistic_model_conf = yaml.load(MODEL_CONF_PATH.open(), Loader=yaml.FullLoader)["model"]
    LATENT_SIZE = 768
    linguistic_model = BenchmarkClassifier(emb_size=40, latent_size=LATENT_SIZE).to('cuda')
    linguistic_model.load_state_dict(torch.load("/home/matterj/codebases/eeg_image_reconstruction/src/trained_models/eeg_img_classification_finetuning_e11_no_Benchmark_v4.pt"))

    

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
    

    dataset_visual = VisualEEGEncoderTrainingDataset(
        latent_image_dataset=latent_image_dataset,
        eeg_dataset=eeg_dataset,
        limit_to_classes=['animal']
    )

    dataset_linguistic = LinguisticEEGEncoderTrainingDataset(
        latent_text_dataset=latent_text_dataset,
        eeg_dataset=eeg_dataset,
        limit_to_classes=['animal']
    )

    IDX = 3
    visual_data = dataset_visual[IDX]
    linguistic_data = dataset_linguistic[IDX]


    decoder = EEGToImageDecoder(
        config=configs,
        visual_decoder=visual_model,
        lingustic_decoder=linguistic_model,
        linguistic_dataset=dataset_linguistic,
        visual_dataset=dataset_visual
    )

    queryier = EEGScienceDirectSQLDBQueryier()
    orig_img = next(queryier.run_query(f"""SELECT 
        image.img_path
        FROM image 
        WHERE image.img_id = '{linguistic_data['img_id']}'
        """))['img']
    orig_img = Image.fromarray(orig_img.astype(np.uint8)).resize((512,512))
    orig_img.save("/home/matterj/codebases/eeg_image_reconstruction/output/decoded/test_decodes/test1_orig.png")
    print(linguistic_data['img_concept'])

    image = decoder.decode(IDX, decode_visual_eeg=False, decode_linguistic_eeg=True)
    image.save("/home/matterj/codebases/eeg_image_reconstruction/output/decoded/test_decodes/test1.png")


    




