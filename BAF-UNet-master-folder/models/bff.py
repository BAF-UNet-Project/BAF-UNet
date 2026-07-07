# import torch
# import torch.nn as nn

# from models.se_block import SEBlock


# class BFF(nn.Module):

#     def __init__(self, low_channels, high_channels, out_channels):
#         super().__init__()

#         self.low_conv = nn.Sequential(
#             nn.Conv2d(low_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.high_conv = nn.Sequential(
#             nn.Conv2d(high_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.boundary_attention = nn.Sequential(
#             nn.Conv2d(out_channels, 1, 1),
#             nn.Sigmoid()
#         )

#         self.se = SEBlock(out_channels)

#     def forward(self, low_feat, high_feat):

#         low_feat = self.low_conv(low_feat)

#         high_feat = self.high_conv(high_feat)

#         high_feat = torch.nn.functional.interpolate(
#             high_feat,
#             size=low_feat.shape[2:],
#             mode='bilinear',
#             align_corners=True
#         )

#         boundary_map = self.boundary_attention(low_feat)

#         reverse_attention = 1 - boundary_map

#         high_feat = high_feat * reverse_attention

#         fused = low_feat + high_feat

#         fused = self.se(fused)

#         return fused





import torch
import torch.nn as nn
import torch.nn.functional as F

from models.se_block import SEBlock


class BFF(nn.Module):
    """
    Boundary-Aware Feature Fusion Module (BFF)
    Exactly follows Section 3.3 and Figure 4 of the paper.
    """

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()

        # 1x1 conv + BN + ReLU for both branches
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention maps (sigmoid)
        self.low_attn = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.high_attn = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Squeeze-and-Excitation
        self.se = SEBlock(out_channels)

    def forward(self, low_feat, high_feat):
        # Project features
        low_feat = self.low_conv(low_feat)
        high_feat = self.high_conv(high_feat)

        # Resize high-level feature to low-level resolution
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(
                high_feat, 
                size=low_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )

        # Generate attention maps
        low_map = self.low_attn(low_feat)      # boundary guidance
        high_map = self.high_attn(high_feat)

        # Reverse attention on high-level feature
        reverse_map = 1.0 - low_map
        high_reweighted = high_feat * reverse_map

        # Complementary low-level attention
        low_reweighted = low_feat * low_map

        # Fuse (paper uses addition of components)
        fused = low_reweighted + high_reweighted + (low_feat * high_map)

        # SE recalibration
        fused = self.se(fused)

        return fused