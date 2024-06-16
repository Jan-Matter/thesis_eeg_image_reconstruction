from torch import nn
import torch

class ContrastiveMSELoss(nn.Module):
    
    def __init__(self, beta=1):
        super(ContrastiveMSELoss, self).__init__()
        self.beta = beta

    def forward(self, output, classes):
        if len(classes.shape) > 1:
            #classes are in binary map format tr
            classes = classes.argmax(dim=1)

        pairwise_mse = torch.mean((output[:, None] - output) ** 2, dim=2)
        pairwise_mse_without_self = pairwise_mse * (1 - torch.eye(output.shape[0]).to(output.device))
        loss_sign = torch.where(classes[:, None] == classes, torch.tensor(1), torch.tensor(-1))
        contrastive_losses = torch.mean(loss_sign * pairwise_mse_without_self, dim=1) + torch.tensor(self.beta)
        contrastive_loss = contrastive_losses.mean()
        return contrastive_loss


if __name__ == '__main__':
    loss = ContrastiveMSELoss(beta=1)
    output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 5.0, 10.0]])
    classes = torch.tensor([1, 0, 1])
    loss = loss(output, classes)
    print(loss)