import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display

from models.baf_unet import BAFUNet


# =========================================================
# CONFIG
# =========================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 512
CHECKPOINT_PATH = 'checkpoints/best_bafunet.pth'

# List of 3 image paths to evaluate
IMAGE_PATHS = [
    '/content/drive/MyDrive/BAF-UNet-master-folder/datasets/PH2DataSet/val_images/IMD008.jpg',
    '/content/drive/MyDrive/BAF-UNet-master-folder/datasets/PH2DataSet/val_images/IMD009.jpg',
    '/content/drive/MyDrive/BAF-UNet-master-folder/datasets/PH2DataSet/val_images/IMD010.jpg'
]


# =========================================================
# LOAD MODEL
# =========================================================

model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint, strict=True)
model.eval()

print("Model loaded successfully!")


# =========================================================
# PREPROCESS BATCH OF 3 IMAGES
# =========================================================

resized_images = []
tensors = []

for path in IMAGE_PATHS:
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_images.append(img_resized)
    
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # [C, H, W]
    tensors.append(img_tensor)

# Stack into a single batch tensor: [3, 3, 512, 512]
batch_tensor = torch.stack(tensors).to(DEVICE)


# =========================================================
# RUN MODEL INFERENCE
# =========================================================

with torch.no_grad():
    logits = model(batch_tensor)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).cpu().numpy().astype(np.uint8)

# Remove channel dimension -> shape: [3, 512, 512]
pred_masks = predictions.squeeze(1)


# =========================================================
# VISUALIZATION (3x3 GRID)
# =========================================================

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

for i in range(3):
    img_resized = resized_images[i]
    pred_mask = pred_masks[i]
    
    # Column 1: Original Image
    axes[i, 0].imshow(img_resized)
    axes[i, 0].set_title(f"Image {i+1}: Original", fontsize=12)
    axes[i, 0].axis('off')
    
    # Column 2: Predicted Mask
    axes[i, 1].imshow(pred_mask, cmap='gray')
    axes[i, 1].set_title(f"Image {i+1}: Predicted Mask", fontsize=12)
    axes[i, 1].axis('off')
    
    # Column 3: Overlay
    axes[i, 2].imshow(img_resized)
    axes[i, 2].imshow(pred_mask, cmap='jet', alpha=0.4)
    axes[i, 2].set_title(f"Image {i+1}: Overlay", fontsize=12)
    axes[i, 2].axis('off')

plt.tight_layout()
output_filename = 'batch_prediction_results.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()

# Display directly if running via !python in Colab terminal
display(IPImage(output_filename))
