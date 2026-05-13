import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from unet.unet_model import UNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Dataset
# Dataset
train_set = BasicDataset(
    images_dir="datasets/ISIC2016/train_images",
    mask_dir="datasets/ISIC2016/train_masks",
    mask_suffix="_Segmentation",
    image_size=384
)

loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)

# Model
model = UNet(n_channels=3, n_classes=2).to(device)

# Loss (simple baseline)
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# One batch
batch = next(iter(loader))
images = batch["image"].to(device)
masks = batch["mask"].to(device)

print("Images:", images.shape)
print("Masks:", masks.shape)

# Forward
outputs = model(images)

print("Outputs:", outputs.shape)

# Compute loss
loss = criterion(outputs, masks)

print("Loss:", loss.item())

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step completed ✅")