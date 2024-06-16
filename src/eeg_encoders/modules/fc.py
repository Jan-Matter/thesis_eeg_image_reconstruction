import torch
from torch import nn
import numpy as np

class FullyConnected(nn.Module):

    def __init__(self, conf, device=torch.device("cpu")):
        super(FullyConnected, self).__init__()
        self.conf = conf
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conf["in_channels"], 
                out_channels=self.conf["out_channels"],
                kernel_size=self.conf["kernel_size"],
                stride=self.conf["stride"],
                padding=self.conf["padding"],
                dilation=self.conf["dilation"]

            ),
            nn.BatchNorm2d(self.conf["out_channels"]),
            nn.ReLU()
        )
        if self.conf["inter_layers"] == 0:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=self.conf["in_features"],
                    out_features=self.conf["out_features"]
                ),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=self.conf["in_features"],
                    out_features=self.conf["inter_features"][0]
                ),
                nn.ReLU(),
                nn.Sequential(*[
                    nn.Sequential(*[
                        nn.Linear(
                            in_features=self.conf["inter_features"][i],
                            out_features=self.conf["inter_features"][i+1]
                        ),
                        nn.ReLU()]
                    )
                    for i in range(self.conf["inter_layers"] - 1)]
                ),
                nn.Linear(
                    in_features=self.conf["inter_features"][self.conf["inter_layers"] - 1],
                    out_features=self.conf["out_features"]
                ),
                #nn.ReLU()
            )
        if self.conf["up_layers"] > 0:
            
            self.up_conv = nn.Sequential(*[
                nn.Sequential(*[
                    nn.ConvTranspose2d(
                        in_channels=self.conf["up_in_channels"][i],
                        out_channels=self.conf["up_out_channels"][i],
                        kernel_size=self.conf["up_kernel_size"][i],
                        stride=self.conf["up_stride"][i],
                        padding=self.conf["up_padding"][i],
                    ),
                    nn.BatchNorm2d(self.conf["up_out_channels"][i]),
                    nn.ReLU()]
                )
                for i in range(self.conf["up_layers"])]
            )
            self.up_conv.to(self.device)
        self.conv.to(self.device)
        self.fc.to(self.device)
    
    def forward(self, x):
        conv_x = self.conv(x)
        fc_x = self.fc(conv_x)
        if self.conf["up_layers"] > 0:
            output_length_shape = int(np.sqrt(fc_x.shape[1] // self.conf["up_in_channels"][0]))
            fc_x = fc_x.reshape(fc_x.shape[0], self.conf["up_in_channels"][0], output_length_shape, output_length_shape)
            up_x = self.up_conv(fc_x)
            output = up_x
        else:
            output = fc_x
        return output
