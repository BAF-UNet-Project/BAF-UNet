import torch


def iou_score(preds, targets, smooth=1e-6):
    """
    Computes IoU (Intersection over Union)
    for binary segmentation.

    Args:
        preds: predicted masks
        targets: ground truth masks

    Returns:
        IoU score
    """

    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum()

    total = preds.sum() + targets.sum()

    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou