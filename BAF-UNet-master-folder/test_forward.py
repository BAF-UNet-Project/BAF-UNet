import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from unet.unet_model import UNet

# Dataset
train_set = BasicDataset(
    images_dir="datasets/ISIC2016/train_images",
    mask_dir="datasets/ISIC2016/train_masks",
    mask_suffix="_Segmentation",
    scale=0.5
)

# DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# Load model
model = UNet(
    n_channels=3,
    n_classes=2,
    bilinear=False
)

# Get one batch
batch = next(iter(train_loader))

images = batch["image"]

print("Input shape:", images.shape)

# Forward pass
outputs = model(images)

print("Output shape:", outputs.shape)