# from torch.utils.data import DataLoader
# from utils.data_loading import BasicDataset

# # Dataset
# train_set = BasicDataset(
#     images_dir="datasets/ISIC2016/train_images",
#     mask_dir="datasets/ISIC2016/train_masks",
#     mask_suffix="_Segmentation"
# )

# # DataLoader
# train_loader = DataLoader(
#     train_set,
#     batch_size=2,
#     shuffle=True,
#     num_workers=0
# )

# # Get first batch
# batch = next(iter(train_loader))

# images = batch["image"]
# masks = batch["mask"]

# print("Batch image shape:", images.shape)
# print("Batch mask shape:", masks.shape)
# print("Image dtype:", images.dtype)
# print("Mask dtype:", masks.dtype)


import albumentations as A
from torch.utils.data import DataLoader

from dataloading import BasicDataset


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5)
])


dataset = BasicDataset(
    images_dir='dataset/train_images',
    mask_dir='dataset/train_masks',
    image_size=512,
    mask_suffix='_segmentation',
    transform=transform
)


loader = DataLoader(dataset, batch_size=2, shuffle=True)


batch = next(iter(loader))

print(batch['image'].shape)
print(batch['mask'].shape)