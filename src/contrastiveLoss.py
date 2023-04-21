import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, gamma=1.2):
        super(ContrastiveLoss, self).__init__()
        self.gamma = gamma

    def probabilidad(self, dif, label):
        return torch.sum(torch.abs(dif-label))/(dif.shape[1])

    def forward(self, output1, output2, label, y):
        dif = output1-output2
        p = self.probabilidad(dif, label)
        loss_contrastive = -self.gamma * y * torch.log(self.probabilidad(dif, label)) - \
                           (1-y) * torch.log(1 - self.probabilidad(dif, label))
        return loss_contrastive
