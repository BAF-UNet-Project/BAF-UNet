import torch
from utils.data_loading import BasicDataset

train_set = BasicDataset(
    images_dir="datasets/ISIC2016/train_images",
    mask_dir="datasets/ISIC2016/train_masks",
    mask_suffix="_Segmentation"
)

test_set = BasicDataset(
    images_dir="datasets/ISIC2016/test_images",
    mask_dir="datasets/ISIC2016/test_masks",
    mask_suffix="_Segmentation"
)

print("Train samples:", len(train_set))
print("Test samples:", len(test_set))