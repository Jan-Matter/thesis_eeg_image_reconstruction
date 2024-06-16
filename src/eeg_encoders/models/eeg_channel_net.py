#This model takes care of the mapping from preprocessed EEG data to the LDM latent Image ingress
#It is based on the implementation of EEG-ChannelNet Architecture described in
# "Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features"

import torch
import yaml
import sys
from pathlib import Path
from torch import nn

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.eeg_encoders.modules.temporal import TemporalConvolution
from src.eeg_encoders.modules.spatial import SpatialConvolution
from src.eeg_encoders.modules.residual import Residual
from src.eeg_encoders.modules.fc import FullyConnected

class EEChannelNet(nn.Module):

    def __init__(self, conf):
        super(EEChannelNet, self).__init__()
        self.device = torch.device(conf["device"])
        self.modules_conf = conf["modules"]
        self.temporal_conv = TemporalConvolution(conf=self.modules_conf["temporal"], device=self.device).to(self.device)
        self.spatial_conv = SpatialConvolution(conf=self.modules_conf["spatial"], device=self.device).to(self.device)
        self.res = Residual(conf=self.modules_conf["residual"], device=self.device).to(self.device)
        self.fc = FullyConnected(conf=self.modules_conf["fc"], device=self.device).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        eeg_channels = x.shape[2]
        x = torch.reshape(x, (batch_size * eeg_channels, 1, x.shape[3]))
        x_temp = self.temporal_conv(x)
        x_spat = self.spatial_conv(x_temp)
        x_spat = torch.reshape(x_spat, (batch_size, x_spat.shape[1], eeg_channels,  x_spat.shape[2]))
        x_res = self.res(x_spat)
        output = self.fc(x_res)
        return output


        
