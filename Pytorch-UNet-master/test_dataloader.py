from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset

# Dataset
train_set = BasicDataset(
    images_dir="datasets/ISIC2016/train_images",
    mask_dir="datasets/ISIC2016/train_masks",
    mask_suffix="_Segmentation"
)

# DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

# Get first batch
batch = next(iter(train_loader))

images = batch["image"]
masks = batch["mask"]

print("Batch image shape:", images.shape)
print("Batch mask shape:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)