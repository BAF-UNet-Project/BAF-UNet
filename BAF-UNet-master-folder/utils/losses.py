# import torch
# import torch.nn as nn


# class DiceLoss(nn.Module):

#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, preds, targets):

#         preds = torch.sigmoid(preds)

#         preds = preds.contiguous().view(-1)
#         targets = targets.contiguous().view(-1).float()

#         intersection = (preds * targets).sum()

#         dice = (
#             2. * intersection + self.smooth
#         ) / (
#             preds.sum() + targets.sum() + self.smooth
#         )

#         return 1 - dice


# class BoundaryAwareLoss(nn.Module):

#     def __init__(self, threshold_epoch=400):
#         super().__init__()

#         self.threshold_epoch = threshold_epoch
#         self.dice = DiceLoss()

#     def forward(self, preds, targets, boundary_map, epoch):

#         base_loss = self.dice(preds, targets)

#         if epoch <= self.threshold_epoch:
#             return base_loss

#         weights = 1 + boundary_map.float()

#         preds = torch.sigmoid(preds)

#         weighted_loss = (
#             weights * (preds - targets.float()) ** 2
#         ).mean()

#         return base_loss + weighted_loss



import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Standard Dice Loss"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1).float()

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class BoundaryAwareLoss(nn.Module):
    """
    Hybrid Loss as described in Section 3.4 of the paper.
    Uses epoch-adaptive boundary-aware weighting on Dice.
    """
    def __init__(self, threshold_epoch=400, boundary_weight=1.0):
        super().__init__()
        self.threshold_epoch = threshold_epoch
        self.boundary_weight = boundary_weight
        self.dice = DiceLoss()

    def forward(self, preds, targets, boundary_maps=None, epoch=0):
        """
        preds: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        boundary_maps: Optional boundary map for weighting
        """
        # Base Dice Loss
        base_loss = self.dice(preds, targets)

        if epoch <= self.threshold_epoch or boundary_maps is None:
            return base_loss

        # Boundary-aware weighted loss (after threshold)
        preds_sig = torch.sigmoid(preds)

        # Expand boundary map if needed
        if len(boundary_maps.shape) == 3:
            boundary_maps = boundary_maps.unsqueeze(1)

        # Weight = 1 + boundary importance
        weights = 1.0 + self.boundary_weight * boundary_maps.float()

        # Weighted Dice (approximated via element-wise weighting)
        weighted_intersection = (weights * preds_sig * targets).sum()
        weighted_pred_sum = (weights * preds_sig).sum()
        weighted_target_sum = (weights * targets).sum()

        weighted_dice = (2. * weighted_intersection + 1e-6) / (
            weighted_pred_sum + weighted_target_sum + 1e-6
        )

        weighted_loss = 1 - weighted_dice

        return base_loss + weighted_loss