import torch
from unet.unet_model import UNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Model
model = UNet(
    n_channels=3,
    n_classes=2,
    bilinear=False
)

# Move model to GPU
model = model.to(device)

# Dummy input
x = torch.randn(1, 3, 256, 256).to(device)

# Forward pass
with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)
print("Output device:", y.device)