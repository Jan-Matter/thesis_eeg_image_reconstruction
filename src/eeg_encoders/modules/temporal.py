import torch
from torch import nn
from torch.nn.parallel import parallel_apply

class TemporalConvolution(nn.Module):

    def __init__(self, conf, device=torch.device("cpu")):
        super(TemporalConvolution, self).__init__()
        self.conf = conf
        self.device = device
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.conf["in_channels"][i], 
                    out_channels=self.conf["out_channels"][i],
                    kernel_size=self.conf["kernel_sizes"][i],
                    stride=self.conf["strides"][i],
                    padding=self.conf["paddings"][i],
                    dilation=self.conf["dilations"][i]
                ),
                nn.BatchNorm1d(self.conf["out_channels"][i]),
                nn.ReLU()
            )
        for i in range(self.conf["conv_count"])])
        for conv in self.convs:
            conv.to(self.device)
    
    def forward(self, x):
        x = x.float()  # Convert input tensor to float
        if self.device.type == "cuda":
            outputs = parallel_apply(self.convs, [x]*len(self.convs))
        else:
            outputs = [conv(x) for conv in self.convs]
        
        out = torch.cat(outputs, dim=1)
        return out


