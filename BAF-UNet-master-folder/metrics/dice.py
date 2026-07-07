# import torch


# def dice_score(preds, targets, smooth=1e-6):
#     """
#     Computes Dice Score for binary segmentation.

#     Args:
#         preds: model predictions (B, H, W)
#         targets: ground truth masks (B, H, W)

#     Returns:
#         Dice score (float tensor)
#     """

#     preds = preds.float()
#     targets = targets.float()

#     intersection = (preds * targets).sum()

#     union = preds.sum() + targets.sum()

#     dice = (2.0 * intersection + smooth) / (union + smooth)

#     return dice




import torch


def dice_score(pred, target, smooth=1e-6):
    """
    pred: [B, H, W] (0 or 1)
    target: [B, H, W] (0 or 1)
    """

    pred = (pred == 1).float()
    target = (target == 1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice