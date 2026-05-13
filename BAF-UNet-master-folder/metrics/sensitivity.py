import torch


def sensitivity_score(preds, targets, smooth=1e-6):
    """
    Computes Sensitivity / Recall.

    Args:
        preds: predicted masks
        targets: ground truth masks

    Returns:
        sensitivity score
    """

    preds = preds.float()
    targets = targets.float()

    tp = (preds * targets).sum()

    fn = ((1 - preds) * targets).sum()

    sensitivity = (tp + smooth) / (tp + fn + smooth)

    return sensitivity