
import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from models.baf_unet import BAFUNet
from utils.metrics import compute_all_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================== CONFIG ======================
IMAGE_SIZE = 512
BATCH_SIZE = 12                    # You can increase this (4~8) if you have enough VRAM
CHECKPOINT_PATH = 'checkpoints/best_bafunet.pth'

# ====================== DATASET ======================
dataset = BasicDataset(
    images_dir='/workspace/datasets/ISIC2017/test_images',
    mask_dir='/workspace/datasets/ISIC2017/test_masks',
    image_size=IMAGE_SIZE,
    mask_suffix='_segmentation',
)

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

# ====================== MODEL ======================
model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE),strict=True)
model.eval()

print(f"Model loaded | Evaluating on {len(dataset)} images with batch_size={BATCH_SIZE}...\n")

# ====================== EVALUATION ======================
all_metrics = []

with torch.no_grad():
    for batch in loader:
        images = batch['image'].to(DEVICE)
        masks = batch['mask'].to(DEVICE).float()

        preds = model(images)
        preds = torch.sigmoid(preds)

        # Compute metrics for each image in batch
        for i in range(preds.shape[0]):
            metric = compute_all_metrics(preds[i:i+1], masks[i:i+1])
            all_metrics.append(metric)

# ====================== AVERAGE ======================
avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) 
               for k in all_metrics[0].keys()}

print("=" * 60)
print("BAF-UNet - ISIC2016 Test Set Results")
print("=" * 60)
print(f"IoU     : {avg_metrics['IoU']:.4f}")
print(f"DSC     : {avg_metrics['DSC']:.4f}")
print(f"Accuracy: {avg_metrics['Acc']:.4f}")
print(f"Sensitivity : {avg_metrics['Sen']:.4f}")
print(f"Specificity : {avg_metrics['Spe']:.4f}")
print("=" * 60)
