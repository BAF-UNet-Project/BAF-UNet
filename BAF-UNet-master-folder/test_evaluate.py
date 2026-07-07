import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from models.unet_parts import models
from evaluate import evaluate

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
test_set = BasicDataset(
    images_dir="datasets/ISIC2016/test_images",
    mask_dir="datasets/ISIC2016/test_masks",
    mask_suffix="_Segmentation",
    image_size=384
)

test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# Model
model = UNet(
    n_channels=3,
    n_classes=2
).to(device)

# Evaluate
results = evaluate(model, test_loader, device)

print("\nEvaluation Results")
print("-------------------")

for k, v in results.items():
    print(f"{k}: {v:.4f}")