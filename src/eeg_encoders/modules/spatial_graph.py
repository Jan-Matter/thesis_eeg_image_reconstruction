from torch_geometric.nn import GCNConv, GATConv
from torch import nn
import torch
from torch.nn.parallel import parallel_apply


class SpatialGraphConvolution(nn.Module):
    def __init__(self, conf, device, gnn_type="gat"):
        self.device = device
        super(SpatialGraphConvolution, self).__init__()
        self.layers = nn.ModuleList()
        time_shape = conf["time_shape"]
        self.gnn_type = gnn_type
        for layer in range(conf["layer_count"]):
            if layer == 0:
                if conf["layer_count"] == 1:
                    gnn_layer = self.__init_gnn_layer(conf["in_channels"], conf["out_channels"], time_shape, gnn_type)
                else:
                    gnn_layer = self.__init_gnn_layer(conf["in_channels"], conf["hidden_channels"][layer], time_shape, gnn_type)
            elif layer != conf["layer_count"] - 1:
                gnn_layer = self.__init_gnn_layer(conf["hidden_channels"][layer - 1], conf["hidden_channels"][layer], time_shape, gnn_type)
            else:
                gnn_layer = self.__init_gnn_layer(conf["hidden_channels"][layer - 1], conf["out_channels"], time_shape, gnn_type)
            self.layers.append(gnn_layer.to(self.device))

    def forward(self, x, edge_index):
        if self.gnn_type == "gat":
            x_out = []
            for x_elem in x:
                for conv in self.layers:
                    x_elem = conv(x_elem, edge_index)
                    x_elem = x_elem.relu()
                x_out.append(x_elem)
            return torch.stack(x_out, dim=0)
        elif self.gnn_type == "gcn":
            for conv in self.layers:
                x = conv(x, edge_index)
                x = x.relu()
            return x
        else:
            raise ValueError("Invalid GNN type")

    def __init_gnn_layer(self, in_channels, out_channels, time_shape, gnn_type):
        if gnn_type == "gat":
            return GATConv(in_channels * time_shape, out_channels * time_shape)
        elif gnn_type == "gcn":
            return GCNConv(in_channels * time_shape, out_channels * time_shape)
        else:
            raise ValueError("Invalid GNN type")

    