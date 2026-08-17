
import os
import gc
import warnings
import logging

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore")

import torch
import albumentations as A
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.data_loading import BasicDataset
from models.baf_unet import BAFUNet
from utils.losses import BoundaryAwareLoss

# ====================== CONFIG ======================
IMAGE_SIZE = 512        # Paper uses 512x512
BATCH_SIZE = 12          # Paper: batch size = 12
NUM_WORKERS = 4 
EPOCHS = 120              # Paper: 600 epochs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoints', exist_ok=True)

# ====================== AUGMENTATION ======================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.2),
])

# ====================== DATASET & LOADER ======================
train_dataset = BasicDataset(
    images_dir='/workspace/datasets/ISIC2017/train_images',
    mask_dir='/workspace/datasets/ISIC2017/train_masks',
    image_size=IMAGE_SIZE,
    mask_suffix='_segmentation',
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

# ====================== MODEL ======================
model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)

# ====================== LOSS ======================
criterion = BoundaryAwareLoss(threshold_epoch=400)   # Paper uses T=400 as best

# ====================== OPTIMIZER & SCHEDULER ======================
optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2
)

# Mixed Precision
scaler = torch.cuda.amp.GradScaler()

best_loss = float('inf')

# ====================== TRAINING ======================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training BAF-UNet on {DEVICE} | Image Size: {IMAGE_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            images = batch['image'].to(DEVICE, non_blocking=True)
            masks = batch['mask'].to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                preds = model(images)                    # (B, 1, H, W)

                loss = criterion(preds.squeeze(1), masks, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/best_bafunet.pth')
            print(">>> Best model saved!")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'checkpoints/bafunet_epoch_{epoch+1}.pth')

        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining Completed!")
