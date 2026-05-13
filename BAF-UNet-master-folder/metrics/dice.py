import torch


def dice_score(preds, targets, smooth=1e-6):
    """
    Computes Dice Score for binary segmentation.

    Args:
        preds: model predictions (B, H, W)
        targets: ground truth masks (B, H, W)

    Returns:
        Dice score (float tensor)
    """

    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice