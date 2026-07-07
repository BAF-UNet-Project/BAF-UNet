# import cv2
# import torch
# import numpy as np

# from PIL import Image

# from models.baf_unet import BAFUNet


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# model = BAFUNet().to(DEVICE)

# model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# model.eval()


# image = Image.open('sample.jpg').convert('RGB')

# image = image.resize((512, 512))

# image = np.array(image).astype(np.float32) / 255.0

# image = image.transpose(2, 0, 1)

# image = torch.tensor(image).unsqueeze(0).float().to(DEVICE)


# with torch.no_grad():

#     pred = model(image)

#     pred = torch.sigmoid(pred)

#     pred = (pred > 0.5).float()


# mask = pred.squeeze().cpu().numpy() * 255

# cv2.imwrite('prediction.png', mask)


import cv2
import torch
import numpy as np
from PIL import Image

from models.baf_unet import BAFUNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================== LOAD MODEL ======================
model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)

model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))
model.eval()

print("Model loaded successfully!")

# ====================== LOAD IMAGE ======================
image_path = 'sample.jpg'

image = Image.open(image_path).convert('RGB')
image = image.resize((512, 512))                    # Must match training size

# Convert to tensor
image_np = np.array(image).astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

# ====================== INFERENCE ======================
with torch.no_grad():
    pred = model(image_tensor)           # Forward pass
    pred = torch.sigmoid(pred)           # Convert to probability
    pred = (pred > 0.5).float()          # Binarize (0 or 1)

# ====================== SAVE MASK ======================
mask = pred.squeeze().cpu().numpy() * 255   # Convert to 0-255 for saving
cv2.imwrite('prediction.png', mask)

print("Prediction saved as 'prediction.png'")