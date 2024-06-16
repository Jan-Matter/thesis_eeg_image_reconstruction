#This model takes care of the mapping from preprocessed EEG data to the LDM latent Image ingress
#It is based on the implementation of EEG-ChannelNet Architecture described in
# "Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features"

import torch
import yaml
import sys
from pathlib import Path
from torch import nn
from braindecode.models import EEGNetv4

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.eeg_encoders.modules.temporal import TemporalConvolution
from src.eeg_encoders.modules.spatial import SpatialConvolution
from src.eeg_encoders.modules.residual import Residual
from src.eeg_encoders.modules.fc import FullyConnected

class VisualEEGNetV4Classifier(nn.Module):

    def __init__(self, conf):
        super(VisualEEGNetV4Classifier, self).__init__()
        self.device = torch.device(conf["device"])
        self.conf = conf
        
        self.model = EEGNetv4(
            n_chans=self.conf["n_chans"],
            n_outputs=self.conf["n_outputs"],
            n_times=self.conf["n_times"],
            final_conv_length='auto',
            pool_mode='mean',
            F1=self.conf["F1"],
            D=self.conf["D"],
            F2=self.conf["F2"],
            kernel_length=self.conf["kernel_length"],
            third_kernel_size=self.conf["third_kernel_size"],
            drop_prob=self.conf["drop_prob"],
            chs_info=None
        ).to(self.device)
        self.linear = nn.Linear(self.conf["n_outputs"], self.conf["out_features"]).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = x.to(torch.float32)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x_eegnet = self.model(x)
        x_out = self.linear(x_eegnet)
        x_out = self.relu(x_out)
        return x_out

if __name__ == "__main__":
    conf_path = Path(__file__).parent.parent.parent / "configs" / "eeg_to_latent_image_eegnetv4_v1.yaml"
    conf = yaml.load(conf_path.open(), Loader=yaml.FullLoader)
    model = VisualEEGNetV4Classifier(conf=conf["model"])
    test_input = {
        "eeg_data_delta": torch.randn(1, 63, 101).to(torch.device("cuda")),
        "eeg_data_theta": torch.randn(1, 63, 101).to(torch.device("cuda")),
        "eeg_data_alpha": torch.randn(1, 63, 101).to(torch.device("cuda")),
        "eeg_data_beta": torch.randn(1, 63, 101).to(torch.device("cuda")),
        "eeg_data_gamma": torch.randn(1, 63, 101).to(torch.device("cuda"))
    }
    output = model(test_input)
    print(output.shape)