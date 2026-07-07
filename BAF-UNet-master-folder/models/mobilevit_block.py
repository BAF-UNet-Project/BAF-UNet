import torch
import torch.nn as nn

from models.transformer import TransformerBlock


class MobileViTBlock(nn.Module):

    def __init__(self, in_channels, dim):
        super().__init__()

        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, dim, 1)
        )

        self.transformer = TransformerBlock(dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.local_rep(x)

        b, c, h, w = y.shape

        y_flat = y.flatten(2).transpose(1, 2)

        y_flat = self.transformer(y_flat)

        y = y_flat.transpose(1, 2).reshape(b, c, h, w)

        out = self.fusion(torch.cat([x, y], dim=1))

        return out