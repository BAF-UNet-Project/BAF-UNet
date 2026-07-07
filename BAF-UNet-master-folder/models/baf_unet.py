# import torch
# import torch.nn as nn

# from models.unet_parts import *
# from models.bavit import BAViT
# from models.bff import BFF


# class BAFUNet(nn.Module):

#     def __init__(self, n_channels=3, n_classes=1):
#         super().__init__()

#         self.inc = DoubleConv(n_channels, 64)

#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)

#         self.bavit1 = BAViT(256)
#         self.bavit2 = BAViT(512)

#         self.bff1 = BFF(512, 1024, 512)
#         self.bff2 = BFF(256, 512, 256)

#         self.up1 = Up(1024 + 512, 512)
#         self.up2 = Up(512 + 256, 256)
#         self.up3 = Up(256 + 128, 128)
#         self.up4 = Up(128 + 64, 64)

#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):

#         x1 = self.inc(x)

#         x2 = self.down1(x1)

#         x3 = self.down2(x2)
#         x3 = self.bavit1(x3)

#         x4 = self.down3(x3)
#         x4 = self.bavit2(x4)

#         x5 = self.down4(x4)

#         f1 = self.bff1(x4, x5)
#         f2 = self.bff2(x3, f1)

#         x = self.up1(x5, f1)
#         x = self.up2(x, f2)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         logits = self.outc(x)

#         return logits





# import torch.nn as nn
# from models.unet_parts import *   # Assume standard U-Net parts
# from models.bavit import BAViT
# from models.bff import BFF

# class BAFUNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=1):
#         super().__init__()

#         self.inc = DoubleConv(n_channels, 64)

#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)

#         # BAViT at selected stages (paper: mid-to-deep levels)
#         self.bavit1 = BAViT(256)
#         self.bavit2 = BAViT(512)

#         # BFF modules for skip connections
#         self.bff1 = BFF(512, 1024, 512)   # deep skip
#         self.bff2 = BFF(256, 512, 256)    # next level

#         self.up1 = Up(1024 + 512, 512)    # Note: input is concat of x5 + bff1
#         self.up2 = Up(512 + 256, 256)
#         self.up3 = Up(256 + 128, 128)
#         self.up4 = Up(128 + 64, 64)

#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)                    # 64

#         x2 = self.down1(x1)                 # 128

#         x3 = self.down2(x2)                 # 256
#         x3 = self.bavit1(x3)

#         x4 = self.down3(x3)                 # 512
#         x4 = self.bavit2(x4)

#         x5 = self.down4(x4)                 # 1024 (bottleneck)

#         # BFF fusion (boundary-aware skips)
#         f1 = self.bff1(x4, x5)              # 512
#         f2 = self.bff2(x3, f1)              # 256   (uses previous fused feature)

#         # Decoder
#         x = self.up1(x5, f1)
#         x = self.up2(x, f2)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         logits = self.outc(x)
#         return logits









import torch
import torch.nn as nn

from models.unet_parts import DoubleConv, Down, Up, OutConv
from models.bavit import BAViT
from models.bff import BFF


class BAFUNet(nn.Module):
    """
    BAF-UNet: Boundary-Aware Feature Fusion UNet
    Fully aligned with the paper "BAF-UNet: a boundary-aware segmentation model for skin lesion segmentation"
    """

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # BAViT blocks at selected stages (as per paper)
        self.bavit1 = BAViT(in_channels=256, dim=128)
        self.bavit2 = BAViT(in_channels=512, dim=128)

        # Boundary-aware Feature Fusion modules
        self.bff1 = BFF(low_channels=512, high_channels=1024, out_channels=512)
        self.bff2 = BFF(low_channels=256, high_channels=512, out_channels=256)

        # Decoder
        self.up1 = Up(1024 + 512, 512)   # x5 + f1
        self.up2 = Up(512 + 256, 256)    # + f2
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)                    # 64

        x2 = self.down1(x1)                 # 128

        x3 = self.down2(x2)                 # 256
        x3 = self.bavit1(x3)

        x4 = self.down3(x3)                 # 512
        x4 = self.bavit2(x4)

        x5 = self.down4(x4)                 # 1024 (bottleneck)

        # BFF Skip Connections (Boundary-aware)
        f1 = self.bff1(x4, x5)              # deep fusion
        f2 = self.bff2(x3, f1)              # next level

        # Decoder
        x = self.up1(x5, f1)
        x = self.up2(x, f2)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits