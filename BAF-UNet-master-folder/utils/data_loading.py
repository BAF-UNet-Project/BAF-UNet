




# import logging
# import numpy as np
# import torch

# from PIL import Image
# from pathlib import Path
# from os import listdir
# from os.path import splitext, isfile, join

# from torch.utils.data import Dataset


# # =========================================================
# # IMAGE LOADER
# # =========================================================

# def load_image(filename):

#     ext = splitext(filename)[1]

#     if ext == '.npy':
#         return Image.fromarray(np.load(filename))

#     elif ext in ['.pt', '.pth']:
#         return Image.fromarray(torch.load(filename).numpy())

#     else:
#         return Image.open(filename)


# # =========================================================
# # FIND UNIQUE MASK VALUES
# # =========================================================

# def unique_mask_values(idx, mask_dir, mask_suffix):

#     mask_file = list(
#         mask_dir.glob(idx + mask_suffix + '.*')
#     )

#     if len(mask_file) == 0:
#         raise FileNotFoundError(
#             f'No mask found for image: {idx}'
#         )

#     mask = np.asarray(load_image(mask_file[0]))

#     if mask.ndim == 2:
#         return np.unique(mask)

#     elif mask.ndim == 3:
#         mask = mask.reshape(-1, mask.shape[-1])
#         return np.unique(mask, axis=0)

#     else:
#         raise ValueError(
#             f'Mask should have 2 or 3 dimensions, got {mask.ndim}'
#         )


# # =========================================================
# # BASIC DATASET
# # =========================================================

# class BasicDataset(Dataset):

#     def __init__(
#         self,
#         images_dir: str,
#         mask_dir: str,
#         image_size: int = 256,
#         mask_suffix: str = '',
#         transform=None
#     ):

#         self.images_dir = Path(images_dir)
#         self.mask_dir = Path(mask_dir)

#         self.image_size = image_size
#         self.mask_suffix = mask_suffix

#         # NEW
#         self.transform = transform

#         # -------------------------------------------------
#         # IMAGE IDS
#         # -------------------------------------------------

#         self.ids = [
#             splitext(file)[0]
#             for file in listdir(images_dir)
#             if isfile(join(images_dir, file))
#             and not file.startswith('.')
#         ]

#         if not self.ids:
#             raise RuntimeError(
#                 f'No input file found in {images_dir}'
#             )

#         logging.info(
#             f'Creating dataset with {len(self.ids)} examples'
#         )

#         # -------------------------------------------------
#         # SCAN MASK VALUES
#         # -------------------------------------------------

#         logging.info('Scanning mask files...')

#         unique = [
#             unique_mask_values(
#                 idx,
#                 self.mask_dir,
#                 self.mask_suffix
#             )
#             for idx in self.ids
#         ]

#         self.mask_values = list(
#             sorted(
#                 np.unique(
#                     np.concatenate(unique),
#                     axis=0
#                 ).tolist()
#             )
#         )

#         logging.info(
#             f'Unique mask values: {self.mask_values}'
#         )

#     # =====================================================
#     # DATASET LENGTH
#     # =====================================================

#     def __len__(self):
#         return len(self.ids)

#     # =====================================================
#     # PREPROCESSING
#     # =====================================================

#     @staticmethod
#     def preprocess(
#         mask_values,
#         pil_img,
#         image_size,
#         is_mask
#     ):

#         # -------------------------------------------------
#         # RESIZE
#         # -------------------------------------------------

#         pil_img = pil_img.resize(
#             (image_size, image_size),
#             resample=Image.NEAREST if is_mask else Image.BICUBIC
#         )

#         img = np.asarray(pil_img)

#         # -------------------------------------------------
#         # MASK PROCESSING
#         # -------------------------------------------------

#         if is_mask:

#             if img.ndim == 3:
#                 img = img[..., 0]

#             img = (img > 0).astype(np.int64)

#             return img

#         # -------------------------------------------------
#         # IMAGE PROCESSING
#         # -------------------------------------------------

#         else:

#             if pil_img.mode != 'RGB':
#                 pil_img = pil_img.convert('RGB')

#             img = np.asarray(pil_img)

#             # Normalize
#             img = img.astype(np.float32) / 255.0

#             return img

#     # =====================================================
#     # GET ITEM
#     # =====================================================

#     def __getitem__(self, idx):

#         name = self.ids[idx]

#         mask_file = list(
#             self.mask_dir.glob(
#                 name + self.mask_suffix + '.*'
#             )
#         )

#         img_file = list(
#             self.images_dir.glob(
#                 name + '.*'
#             )
#         )

#         if len(img_file) != 1:
#             raise RuntimeError(
#                 f'Image issue for ID: {name}'
#             )

#         if len(mask_file) != 1:
#             raise RuntimeError(
#                 f'Mask issue for ID: {name}'
#             )

#         mask = load_image(mask_file[0])
#         img = load_image(img_file[0])

#         if img.size != mask.size:
#             raise RuntimeError(
#                 f'Size mismatch for {name}: '
#                 f'Image={img.size}, Mask={mask.size}'
#             )

#         # -------------------------------------------------
#         # PREPROCESS
#         # -------------------------------------------------

#         img = self.preprocess(
#             self.mask_values,
#             img,
#             self.image_size,
#             is_mask=False
#         )

#         mask = self.preprocess(
#             self.mask_values,
#             mask,
#             self.image_size,
#             is_mask=True
#         )

#         # =================================================
#         # AUGMENTATION
#         # =================================================

#         if self.transform is not None:

#             transformed = self.transform(
#                 image=img,
#                 mask=mask
#             )

#             img = transformed['image']
#             mask = transformed['mask']

#         # =================================================
#         # HWC -> CHW
#         # =================================================

#         img = img.transpose((2, 0, 1))

#         return {
#             'image': torch.as_tensor(img.copy()).float().contiguous(),
#             'mask': torch.as_tensor(mask.copy()).long().contiguous()
#         }


# # =========================================================
# # CARVANA DATASET
# # =========================================================

# class CarvanaDataset(BasicDataset):

#     def __init__(
#         self,
#         images_dir,
#         mask_dir,
#         image_size=512,
#         transform=None
#     ):

#         super().__init__(
#             images_dir=images_dir,
#             mask_dir=mask_dir,
#             image_size=image_size,
#             mask_suffix='_Segmentation',
#             transform=transform
#         )




import logging
import numpy as np
import torch

from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join

from torch.utils.data import Dataset
from PIL import Image


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        image_size: int = 512,
        mask_suffix: str = '_Segmentation',
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.transform = transform

        # Get image IDs
        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]

        if not self.ids:
            raise RuntimeError(f'No images found in {images_dir}')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Scan unique mask values (for multi-class, but we use binary)
        logging.info('Scanning mask files...')
        unique = []
        for idx in self.ids:
            mask_file = list(self.mask_dir.glob(idx + mask_suffix + '.*'))
            if mask_file:
                mask = np.asarray(Image.open(mask_file[0]))
                unique.append(np.unique(mask))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique)).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, image_size, is_mask):
        """Preprocess image or mask"""
        pil_img = pil_img.resize((image_size, image_size),
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)

        img = np.asarray(pil_img)

        if is_mask:
            if img.ndim == 3:
                img = img[..., 0]
            img = (img > 0).astype(np.int64)
            return img
        else:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.asarray(pil_img).astype(np.float32) / 255.0
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Find files
        img_file = list(self.images_dir.glob(name + '.*'))[0]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))[0]

        img = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file)

        # Preprocess
        img = self.preprocess(img, self.image_size, is_mask=False)
        mask = self.preprocess(mask, self.image_size, is_mask=True)

        # Apply Albumentations transform
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # Convert to tensor
        img = img.transpose((2, 0, 1))   # HWC -> CHW

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


# Optional: For Carvana-style datasets
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, image_size=512, transform=None):
        super().__init__(
            images_dir=images_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            mask_suffix='_Segmentation',
            transform=transform
        )