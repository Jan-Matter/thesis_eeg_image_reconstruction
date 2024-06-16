import torch
from torch import nn
from torch.nn.parallel import parallel_apply

class Residual(nn.Module):

    def __init__(self, conf, device=torch.device("cpu")):
        super(Residual, self).__init__()
        self.conf = conf
        self.device = device
        self.layers = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(
                        in_channels=self.conf["in_channels"][i][j],
                        out_channels=self.conf["out_channels"][i][j],
                        kernel_size=self.conf["kernel_sizes"][i][j],
                        stride=self.conf["strides"][i][j],
                        padding=self.conf["paddings"][i][j],
                        dilation=self.conf["dilations"][i][j]
                        ),
                        nn.BatchNorm2d(self.conf["out_channels"][i][j]),
                        nn.ReLU(),
                        nn.AvgPool2d(self.conf["pool_sizes"][i][j])
                    )
                for j in range(self.conf["conv_count"])])
            for i in range(self.conf["layer_count"])
        ])
        for layer in self.layers:
            for conv in layer:
                conv.to(self.device)

        
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x_res = nn.AvgPool2d(self.conf["input_pool_size"][i])(x)
            for conv in layer:
                x = conv(x)
            x = x + x_res #add residual connection
        return x
            

