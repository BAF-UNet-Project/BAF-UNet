# import torch


# def specificity(preds, targets, smooth=1e-6):
#     """
#     Specificity = TN / (TN + FP)
#     """

#     preds = preds.view(-1)
#     targets = targets.view(-1)

#     tn = ((preds == 0) & (targets == 0)).float().sum()
#     fp = ((preds == 1) & (targets == 0)).float().sum()

#     return (tn + smooth) / (tn + fp + smooth)


# import torch


# def sensitivity_score(pred, target, smooth=1e-6):

#     pred = (pred == 1)
#     target = (target == 1)

#     tp = (pred & target).sum()
#     fn = (~pred & target).sum()

#     return (tp + smooth) / (tp + fn + smooth)


import torch


def specificity_score(pred, target, smooth=1e-6):

    pred = (pred == 1)
    target = (target == 1)

    tn = (~pred & ~target).sum()
    fp = (pred & ~target).sum()

    return (tn + smooth) / (tn + fp + smooth)