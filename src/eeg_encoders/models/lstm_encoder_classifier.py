import torch
import yaml
import sys
from pathlib import Path
from torch import nn

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

class LSTMEncoderClassifier(nn.Module):

    def __init__(self, conf):
        super(LSTMEncoderClassifier, self).__init__()
        self.device = torch.device(conf["device"])
        self.modules_conf = conf["modules"]
        self.lstm_layers = nn.LSTM(
            self.modules_conf['lstm']['in_channels'],
            self.modules_conf['lstm']['out_channels'],
            num_layers=self.modules_conf['lstm']['layer_count'],
            batch_first=True,
            dropout=self.modules_conf['lstm']['dropout']
        ).to(self.device)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.modules_conf['lstm']['out_channels'],
                self.modules_conf['fc']['in_features']
            ),
            nn.ReLU(),
            nn.Linear(
                self.modules_conf['fc']['in_features'],
                self.modules_conf['fc']['out_features']
            ),
            nn.ReLU()
            ).to(self.device)
            
    
    def forward(self, x):
        x = x.float()
        x = x.to(self.device)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        x_lstm = self.lstm_layers(x)[0][:,-1,:] # get the last output
        x_out = self.fc(x_lstm)
        return x_out
    
        
