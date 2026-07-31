"""Composite BCE + Dice loss (paper eq. 9): L_seg = L_BCE + L_Dice.

Bug fixed vs. the original notebook, which trained with plain BCELoss() only —
no explicit region-overlap term, which hurts on class-imbalanced foreground
(nuclei/gland pixels are a small minority of each image).
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + self.eps) / (pred.sum(dim=1) + target.sum(dim=1) + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice(pred, target)
