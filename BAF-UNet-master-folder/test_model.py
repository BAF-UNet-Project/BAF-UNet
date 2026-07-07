# import torch

# from models.baf_unet import BAFUNet


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = BAFUNet().to(DEVICE)

# x = torch.randn(2, 3, 512, 512).to(DEVICE)

# y = model(x)

# print("Input shape :", x.shape)
# print("Output shape:", y.shape)


# import torch

# from models.baf_unet import BAFUNet

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = BAFUNet().to(DEVICE)

# # SMALLER INPUT
# x = torch.randn(8, 3, 512,512).to(DEVICE)
# model.eval()  # Set model to evaluation mode
# with torch.no_grad():
#     y = model(x)

# print("Input shape :", x.shape)
# print("Output shape:", y.shape)


import torch

from models.baf_unet import BAFUNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================== MODEL ======================
model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)

# ====================== TEST ======================
model.eval()   # Important: evaluation mode

# Create dummy input (batch_size=8, 3 channels, 512x512)
x = torch.randn(2, 3, 512, 512).to(DEVICE)

with torch.no_grad():        # No gradient calculation (faster + less memory)
    y = model(x)

print("Input shape  :", x.shape)
print("Output shape :", y.shape)
print("Test passed!" if y.shape == (2, 1, 512, 512) else "❌ Shape mismatch!")
