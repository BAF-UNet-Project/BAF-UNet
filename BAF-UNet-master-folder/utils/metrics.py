# import torch


# SMOOTH = 1e-6


# def get_stats(pred, mask):

#     pred = (pred > 0.5).float()
#     mask = mask.float()

#     TP = (pred * mask).sum()

#     FP = (pred * (1 - mask)).sum()

#     FN = ((1 - pred) * mask).sum()

#     TN = ((1 - pred) * (1 - mask)).sum()

#     return TP, FP, FN, TN


# # IoU

# def iou_score(pred, mask):

#     TP, FP, FN, TN = get_stats(pred, mask)

#     return ((TP + SMOOTH) / (TP + FP + FN + SMOOTH)).item()


# # Dice

# def dice_score(pred, mask):

#     TP, FP, FN, TN = get_stats(pred, mask)

#     return ((2 * TP + SMOOTH) /
#             (2 * TP + FP + FN + SMOOTH)).item()


# # Accuracy

# def accuracy(pred, mask):

#     TP, FP, FN, TN = get_stats(pred, mask)

#     return ((TP + TN + SMOOTH) /
#             (TP + TN + FP + FN + SMOOTH)).item()


# # Sensitivity

# def sensitivity(pred, mask):

#     TP, FP, FN, TN = get_stats(pred, mask)

#     return ((TP + SMOOTH) /
#             (TP + FN + SMOOTH)).item()


# # Specificity

# def specificity(pred, mask):

#     TP, FP, FN, TN = get_stats(pred, mask)

#     return ((TN + SMOOTH) /
#             (TN + FP + SMOOTH)).item()










import torch

SMOOTH = 1e-6


def get_stats(pred, mask):
    """Compute TP, FP, FN, TN"""
    pred = (pred > 0.5).float()
    mask = mask.float()

    TP = (pred * mask).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()
    TN = ((1 - pred) * (1 - mask)).sum()

    return TP, FP, FN, TN


def iou_score(pred, mask):
    """Intersection over Union (IoU)"""
    TP, FP, FN, TN = get_stats(pred, mask)
    return ((TP + SMOOTH) / (TP + FP + FN + SMOOTH)).item()


def dice_score(pred, mask):
    """Dice Similarity Coefficient (DSC)"""
    TP, FP, FN, TN = get_stats(pred, mask)
    return ((2 * TP + SMOOTH) / (2 * TP + FP + FN + SMOOTH)).item()


def accuracy(pred, mask):
    """Accuracy (Acc)"""
    TP, FP, FN, TN = get_stats(pred, mask)
    return ((TP + TN + SMOOTH) / (TP + TN + FP + FN + SMOOTH)).item()


def sensitivity(pred, mask):
    """Sensitivity / Recall (Sen)"""
    TP, FP, FN, TN = get_stats(pred, mask)
    return ((TP + SMOOTH) / (TP + FN + SMOOTH)).item()


def specificity(pred, mask):
    """Specificity (Spe)"""
    TP, FP, FN, TN = get_stats(pred, mask)
    return ((TN + SMOOTH) / (TN + FP + SMOOTH)).item()


# Optional: Return all metrics at once
def compute_all_metrics(pred, mask):
    """Return all metrics as dict (useful for validation)"""
    return {
        'IoU': iou_score(pred, mask),
        'DSC': dice_score(pred, mask),
        'Acc': accuracy(pred, mask),
        'Sen': sensitivity(pred, mask),
        'Spe': specificity(pred, mask),
    }