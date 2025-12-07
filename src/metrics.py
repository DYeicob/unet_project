import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def iou_score(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float().sum()
    return correct / y_true.numel()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, size_average=size_average)
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)
