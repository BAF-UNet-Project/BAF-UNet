# import torch
# import torch.nn as nn

# from models.transformer import TransformerBlock


# class BoundaryAttention(nn.Module):

#     def __init__(self, channels):
#         super().__init__()

#         self.att = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.att(x)


# class BAViT(nn.Module):

#     def __init__(self, in_channels, dim=128):
#         super().__init__()

#         self.local_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.boundary_attention = BoundaryAttention(in_channels)

#         self.to_transformer = nn.Conv2d(in_channels, dim, 1)

#         self.transformer = TransformerBlock(dim)

#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(dim, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.final_fusion = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):

#         local_feat = self.local_conv(x)

#         boundary_map = self.boundary_attention(local_feat)

#         boundary_feat = local_feat * boundary_map

#         t = self.to_transformer(boundary_feat)

#         b, c, h, w = t.shape

#         t = t.flatten(2).transpose(1, 2)

#         t = self.transformer(t)

#         t = t.transpose(1, 2).reshape(b, c, h, w)

#         global_feat = self.conv3x3(t)

#         out = self.final_fusion(
#             torch.cat([global_feat, x], dim=1)
#         )

#         return out

import torch
import torch.nn as nn

from models.transformer import TransformerBlock


class BoundaryAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.att(x)


class BAViT(nn.Module):
    """
    Boundary-Aware ViT (BAViT) - Extends MobileViT as per paper Section 3.2
    """
    def __init__(self, in_channels, dim=128):
        super().__init__()

        # Local representation (similar to MobileViT)
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),   # extra capacity
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.boundary_attention = BoundaryAttention(in_channels)

        # Project to transformer dim
        self.to_transformer = nn.Conv2d(in_channels, dim, kernel_size=1)

        self.transformer = TransformerBlock(dim=dim, heads=4, mlp_dim=dim*4)

        # Global conv
        self.conv_global = nn.Sequential(
            nn.Conv2d(dim, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion (Eq. 5)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Local + Boundary
        local_feat = self.local_rep(x)
        boundary_map = self.boundary_attention(local_feat)
        boundary_enhanced = local_feat * boundary_map

        # Transformer branch
        t = self.to_transformer(boundary_enhanced)
        b, c, h, w = t.shape
        t_flat = t.flatten(2).transpose(1, 2)
        t_trans = self.transformer(t_flat)
        t_trans = t_trans.transpose(1, 2).reshape(b, c, h, w)

        global_feat = self.conv_global(t_trans)

        # Final fusion with original input
        out = self.fusion(torch.cat([global_feat, x], dim=1))
        return out