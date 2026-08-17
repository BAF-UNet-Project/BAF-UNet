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

    Fix vs. original implementation:
    The transformer branch previously flattened the ENTIRE feature map
    (e.g. 128x128 = 16384 tokens) into a single sequence and ran full,
    unwindowed multi-head self-attention across it. That is O(N^2) memory
    (~40-50GB at the paper's batch size) and OOMs on consumer GPUs.

    This version restores the MobileViT-style "unfold into local patches ->
    attend within each patch -> fold back" step (Eq. referenced as
    to_transformer -> flatten in the spec) via `patch_size`. Attention is
    now computed within non-overlapping p x p patches instead of globally,
    which reduces attention-matrix memory by a factor of num_patches
    (256x reduction at patch_size=8 on a 128x128 map) with no changes
    required inside TransformerBlock itself.
    """

    def __init__(self, in_channels, dim=128, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        # Local representation (similar to MobileViT)
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # extra capacity
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.boundary_attention = BoundaryAttention(in_channels)

        # Project to transformer dim
        self.to_transformer = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.transformer = TransformerBlock(dim=dim, heads=4, mlp_dim=dim * 2)

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

    def _unfold_to_patches(self, t):
        """
        (b, c, h, w) -> (b * num_patches, p*p, c)
        Splits the feature map into non-overlapping p x p patches and
        stacks them along the batch dimension so the transformer attends
        only within each patch, not across the whole map.
        """
        b, c, h, w = t.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, (
            f"Feature map size (h={h}, w={w}) must be divisible by "
            f"patch_size={p}."
        )

        t = t.reshape(b, c, h // p, p, w // p, p)      # b, c, h/p, p, w/p, p
        t = t.permute(0, 2, 4, 3, 5, 1)                 # b, h/p, w/p, p, p, c
        t = t.reshape(b * (h // p) * (w // p), p * p, c)  # (b*num_patches, p*p, c)
        return t, (b, c, h, w)

    def _fold_from_patches(self, t, shape):
        """
        Inverse of _unfold_to_patches:
        (b * num_patches, p*p, c) -> (b, c, h, w)
        """
        b, c, h, w = shape
        p = self.patch_size

        t = t.reshape(b, h // p, w // p, p, p, c)       # b, h/p, w/p, p, p, c
        t = t.permute(0, 5, 1, 3, 2, 4)                  # b, c, h/p, p, w/p, p
        t = t.reshape(b, c, h, w)
        return t

    def forward(self, x):
        # Local + Boundary
        local_feat = self.local_rep(x)
        boundary_map = self.boundary_attention(local_feat)
        boundary_enhanced = local_feat * boundary_map

        # Transformer branch (now windowed into patches)
        t = self.to_transformer(boundary_enhanced)       # b, dim, h, w

        t_patches, shape = self._unfold_to_patches(t)     # (b*num_patches, p*p, dim)
        t_trans = self.transformer(t_patches)              # attention within each patch only
        t_trans = self._fold_from_patches(t_trans, shape)  # back to b, dim, h, w

        global_feat = self.conv_global(t_trans)

        # Final fusion with original input
        out = self.fusion(torch.cat([global_feat, x], dim=1))
        return out
