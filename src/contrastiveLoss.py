import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, gamma=1.2, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.gamma = gamma
        self.margin = margin

    def probabilidad(self, dif, label):
        return torch.sum(torch.abs(dif-label))/(3*dif.shape[1])

    def forward(self, output1, output2, label, y):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = (1-y) * torch.pow(euclidean_distance, 2) + \
                           self.gamma * y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_contrastive)

class ContrastiveLossFF(torch.nn.Module):
    def __init__(self, gamma=1.2, margin=2):
        super(ContrastiveLossFF, self).__init__()
        self.gamma = gamma
        self.margin = margin
        self.fcn = nn.Linear(3, 3)

    def forward(self, output1, output2, label, y):
        dif = output1-output2
        contrast = self.fcn(dif.type(torch.FloatTensor))
        if torch.isnan(torch.log(1 - self.probabilidad(contrast, label))):
            loss_contrastive = -self.gamma * y * torch.log(self.probabilidad(contrast, label))
        elif torch.isnan(torch.log(self.probabilidad(contrast, label))):
            loss_contrastive = - (1-y) * torch.log(1 - self.probabilidad(contrast, label))
        else:
            loss_contrastive = -self.gamma * y * torch.log(self.probabilidad(contrast, label)) - \
                               (1-y) * torch.log(1 - self.probabilidad(contrast, label))
        return loss_contrastive
