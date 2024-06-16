import torch
import yaml
import sys
from pathlib import Path
from torch import nn

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.eeg_encoders.modules.temporal import TemporalConvolution
from src.eeg_encoders.modules.spatial_attention import SpatialAttention
from src.eeg_encoders.modules.residual import Residual
from src.eeg_encoders.modules.fc import FullyConnected

class AttentionEncoderClassifier(nn.Module):

    def __init__(self, conf):
        super(AttentionEncoderClassifier, self).__init__()
        self.device = torch.device(conf["device"])
        self.modules_conf = conf["modules"]

        self.temporal_conv = TemporalConvolution(conf=self.modules_conf["temporal"], device=self.device).to(self.device)
        self.spatial_attention = SpatialAttention(
            conf=self.modules_conf["spatial"],
            device=self.device).to(self.device)
        self.res = Residual(conf=self.modules_conf["residual"], device=self.device).to(self.device)
        self.fc = FullyConnected(conf=self.modules_conf["fc"], device=self.device).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        eeg_channels = x.shape[2]
        time_shape = x.shape[3]
        x = torch.reshape(x, (batch_size * eeg_channels, 1, x.shape[3]))
        x_temp = self.temporal_conv(x)
        x_temp = torch.reshape(x_temp, (batch_size, eeg_channels, x_temp.shape[1] * x_temp.shape[2]))
        x_spat = self.spatial_attention(x_temp)
        x_spat = torch.reshape(x_spat, (batch_size, x_spat.shape[2] // time_shape, eeg_channels, time_shape))
        x_res = self.res(x_spat)
        output = self.fc(x_res)
        return output


