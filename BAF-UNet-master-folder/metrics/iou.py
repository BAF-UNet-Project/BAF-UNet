import torch


def iou_score(pred, target, smooth=1e-6):

    pred = (pred == 1).float()
    target = (target == 1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)