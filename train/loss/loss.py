
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np



class LossWithWeights(nn.Module):
    def __init__(self):
        super(LossWithWeights, self).__init__()

    @abstractmethod
    def weight_from_iter(self, iter_index):
        pass


class L1Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L1Loss(x, y)

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class L2Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L2Loss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L2Loss(x, y)

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class CrossEntropyLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.ce_loss(x, y)

    def weight_from_iter(self, iter_index):
        return self.loss_weight
