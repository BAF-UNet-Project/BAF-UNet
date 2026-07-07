# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # =========================================================
# # DICE LOSS (MULTI-CLASS SAFE)
# # =========================================================
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, preds, targets):
#         """
#         preds: [B, C, H, W] logits
#         targets: [B, H, W]
#         """

#         preds = torch.softmax(preds, dim=1)

#         # one-hot encode targets
#         targets = F.one_hot(targets, num_classes=preds.shape[1])
#         targets = targets.permute(0, 3, 1, 2).float()

#         intersection = (preds * targets).sum(dim=(0, 2, 3))
#         union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))

#         dice = (2. * intersection + self.smooth) / (union + self.smooth)

#         return 1 - dice.mean()


# # =========================================================
# # HYBRID LOSS (CE + DICE)
# # =========================================================
# class HybridLoss(nn.Module):
#     def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
#         super().__init__()

#         # IMPORTANT: handles class imbalance (fixes background dominance)
#         self.ce = nn.CrossEntropyLoss(weight=class_weights)

#         self.dice = DiceLoss()
#         self.ce_w = ce_weight
#         self.dice_w = dice_weight

#     def forward(self, preds, targets):
#         ce_loss = self.ce(preds, targets)
#         dice_loss = self.dice(preds, targets)

#         return self.ce_w * ce_loss + self.dice_w * dice_loss



import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        preds = probs[:, 1, :, :]  # lesion class

        targets = (targets == 1).float()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class HybridLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.ce_loss = WeightedCrossEntropy(class_weights)
        self.dice_loss = DiceLoss()

        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.ce_w * ce + self.dice_w * dice