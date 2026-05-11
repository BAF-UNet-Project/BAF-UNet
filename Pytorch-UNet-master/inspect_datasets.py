# import matplotlib
# matplotlib.use("TkAgg")

# import matplotlib.pyplot as plt
# from utils.data_loading import BasicDataset

# # Load dataset
# train_set = BasicDataset(
#     images_dir="datasets/ISIC2016/train_images",
#     mask_dir="datasets/ISIC2016/train_masks",
#     mask_suffix="_Segmentation"
# )

# sample = train_set[0]

# image = sample["image"]
# mask = sample["mask"]

# print("Image shape:", image.shape)
# print("Mask shape:", mask.shape)
# print("Mask unique values:", mask.unique())

# image_np = image.permute(1, 2, 0).numpy()

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image_np)
# plt.title("Image")

# plt.subplot(1, 2, 2)
# plt.imshow(mask.numpy(), cmap="gray")
# plt.title("Mask")

# plt.show()




import matplotlib.pyplot as plt
from utils.data_loading import BasicDataset

# Load dataset
train_set = BasicDataset(
    images_dir="datasets/ISIC2016/train_images",
    mask_dir="datasets/ISIC2016/train_masks",
    mask_suffix="_Segmentation"
)

# Get first sample
sample = train_set[0]

image = sample["image"]
mask = sample["mask"]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", mask.unique())

# Convert image for visualization
image_np = image.permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(mask.numpy(), cmap="gray")
plt.title("Mask")

plt.show()