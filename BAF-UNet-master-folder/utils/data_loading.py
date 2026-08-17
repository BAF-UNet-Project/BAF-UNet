import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
    MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        image_size: int = 256,
        mask_suffix: str = "_segmentation",
        transform=None
    ):
        super().__init__()

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.transform = transform

        if not self.images_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError("Image or Mask directory does not exist.")

        # 1. Fetch input images
        image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS and not f.name.startswith(".")
        ])

        if not image_files:
            raise RuntimeError(f"No JPEG images found in {self.images_dir}")

        # 2. Build a Mask Map ONCE (O(N) instead of O(N^2))
        mask_map = {
            f.stem: f for f in self.mask_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.MASK_EXTENSIONS
        }

        # 3. Match images with masks instantaneously in memory
        self.samples = []
        self.missing_masks = []

        for image_file in image_files:
            image_id = image_file.stem
            expected_mask_name = image_id + self.mask_suffix

            if expected_mask_name in mask_map:
                self.samples.append({
                    "id": image_id,
                    "image": image_file,
                    "mask": mask_map[expected_mask_name]
                })
            else:
                self.missing_masks.append(image_file.name)

        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs matched with suffix '{self.mask_suffix}'.")

        self.ids = [sample["id"] for sample in self.samples]
        self.mask_values = [0, 1]  # Standard binary mask values

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def preprocess(pil_img, image_size, is_mask):
        pil_img = pil_img.resize(
            (image_size, image_size),
            resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        )

        if is_mask:
            mask = np.asarray(pil_img)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return (mask > 0).astype(np.int64)

        else:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image = np.asarray(pil_img).astype(np.float32)
            return image / 255.0

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample["image"]) as img:
            image = img.convert("RGB").copy()

        with Image.open(sample["mask"]) as msk:
            mask = msk.copy()

        image = self.preprocess(image, self.image_size, is_mask=False)
        mask = self.preprocess(mask, self.image_size, is_mask=True)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image.contiguous(),
            "mask": mask.contiguous()
        }
