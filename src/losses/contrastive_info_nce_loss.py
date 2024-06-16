from torch import nn
import numpy as np
import torch

class ContrastiveInfoNCELoss(nn.Module):
    
    def __init__(self, temperature=1):
        super(ContrastiveInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, encoded_eeg, encoded_img):
        batch_size = encoded_eeg.shape[0]
        encoded_eeg = encoded_eeg.double()
        encoded_img = encoded_img.double()
        encoded_eeg = encoded_eeg / encoded_eeg.norm(dim=1, keepdim=True)
        encoded_img = encoded_img / encoded_img.norm(dim=1, keepdim=True)
        logits = torch.mm(encoded_eeg, encoded_img.t()) * np.exp(self.temperature)
        labels = torch.arange(batch_size).to(logits.device)
        loss_e = nn.CrossEntropyLoss()(logits, labels)
        loss_i = nn.CrossEntropyLoss()(logits.t(), labels)
        loss = (loss_e + loss_i) / 2
        return loss


if __name__ == '__main__':
    loss = ContrastiveInfoNCELoss(temperature=1)
    encoded_eeg = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 5.0, 10.0]])
    encoded_img = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 5.0, 10.0]])
    loss = loss(encoded_eeg, encoded_img)
    print(loss)