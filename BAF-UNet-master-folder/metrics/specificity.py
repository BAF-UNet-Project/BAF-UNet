import torch


def specificity(preds, targets, smooth=1e-6):
    """
    Specificity = TN / (TN + FP)
    """

    preds = preds.view(-1)
    targets = targets.view(-1)

    tn = ((preds == 0) & (targets == 0)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()

    return (tn + smooth) / (tn + fp + smooth)