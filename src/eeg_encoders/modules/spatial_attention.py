from torch import nn
import torch
from torch.nn.parallel import parallel_apply


class SpatialAttention(nn.Module):
    def __init__(self, conf, device):
        self.device = device
        super(SpatialAttention, self).__init__()
        self.layers = nn.ModuleList()
        time_shape = conf["time_shape"]
        for layer in range(conf["layer_count"]):
            attention_layer = self.__init_attention_layer(conf["channels"], conf["num_heads"], time_shape)
            self.layers.append(attention_layer.to(self.device))

    def forward(self, x):
        for layer in self.layers:
           x = layer(x, x, x)[0]
        return x

    def __init_attention_layer(self, channels, num_heads, time_shape):
        if channels * time_shape % num_heads != 0:
            raise ValueError("The number of output channels must be a multiple of the number of input channels.")
        attention_layer = nn.MultiheadAttention(channels * time_shape, num_heads, batch_first=True)
        return attention_layer