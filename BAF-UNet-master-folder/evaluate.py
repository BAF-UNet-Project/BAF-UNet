# # import torch
# # from tqdm import tqdm

# # from metrics.dice import dice_score
# # from metrics.iou import iou_score
# # from metrics.accuracy import accuracy_score
# # from metrics.sensitivity import sensitivity_score
# # from metrics.specificity import specificity


# # @torch.inference_mode()
# # def evaluate(model, dataloader, device):
# #     model.eval()

# #     dice_total = 0
# #     iou_total = 0
# #     acc_total = 0
# #     sens_total = 0
# #     spec_total = 0

# #     num_batches = len(dataloader)

# #     for batch in tqdm(dataloader, desc="Evaluating"):
# #         images = batch["image"].to(device)
# #         masks = batch["mask"].to(device)

# #         # Forward
# #         outputs = model(images)

# #         # Convert logits -> probabilities
# #         # probs = torch.softmax(outputs, dim=1)

# #         # # Get predicted class
# #         # preds = torch.argmax(probs, dim=1)
# #         probs = torch.softmax(outputs, dim=1)
# #         preds = torch.argmax(probs, dim=1)


# #         # Metrics
# #         dice_total += dice_score(preds, masks).item()
# #         iou_total += iou_score(preds, masks).item()
# #         acc_total += accuracy_score(preds, masks).item()
# #         sens_total += sensitivity_score(preds, masks).item()
# #         spec_total += specificity(preds, masks).item()

# #     results = {
# #         "Dice": dice_total / num_batches,
# #         "IoU": iou_total / num_batches,
# #         "Accuracy": acc_total / num_batches,
# #         "Sensitivity": sens_total / num_batches,
# #         "Specificity": spec_total / num_batches
# #     }

# #     return results


# # import torch
# # from tqdm import tqdm

# # from metrics.dice import dice_score
# # from metrics.iou import iou_score
# # from metrics.accuracy import accuracy_score
# # from metrics.sensitivity import sensitivity_score
# # from metrics.specificity import specificity_score   # FIXED NAME


# # @torch.inference_mode()
# # def evaluate(model, dataloader, device):
# #     model.eval()

# #     dice_total = 0
# #     iou_total = 0
# #     acc_total = 0
# #     sens_total = 0
# #     spec_total = 0

# #     num_batches = len(dataloader)

# #     for batch in tqdm(dataloader, desc="Evaluating"):
# #         images = batch["image"].to(device)
# #         masks = batch["mask"].to(device).long()

# #         outputs = model(images)

# #         # ✅ FIXED PREDICTION PIPELINE
# #         preds = torch.argmax(outputs, dim=1)

# #         # Ensure same type
# #         preds = preds.long()
# #         masks = masks.long()

# #         # Metrics (ALL CLASS-BASED)
# #         dice_total += dice_score(preds, masks).item()
# #         iou_total += iou_score(preds, masks).item()
# #         acc_total += accuracy_score(preds, masks).item()
# #         sens_total += sensitivity_score(preds, masks).item()
# #         spec_total += specificity_score(preds, masks).item()

# #     results = {
# #         "Dice": dice_total / num_batches,
# #         "IoU": iou_total / num_batches,
# #         "Accuracy": acc_total / num_batches,
# #         "Sensitivity": sens_total / num_batches,
# #         "Specificity": spec_total / num_batches
# #     }

# #     return results




# import torch

# from torch.utils.data import DataLoader

# from utils.data_loading import BasicDataset
# from models.baf_unet import BAFUNet
# from utils.metrics import *


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# dataset = BasicDataset(
#     images_dir='datasets/ISIC2016/test_images',
#     mask_dir='datasets/ISIC2016/test_masks',
#     image_size=512,
#     mask_suffix='_segmentation'
# )


# loader = DataLoader(dataset, batch_size=1)


# model = BAFUNet().to(DEVICE)

# model.load_state_dict(
#     torch.load('checkpoints/best_model.pth')
# )

# model.eval()

# ious = []
# dices = []
# accs = []
# sens = []
# spes = []


# with torch.no_grad():

#     for batch in loader:

#         image = batch['image'].to(DEVICE)
#         mask = batch['mask'].to(DEVICE)

#         pred = model(image)

#         pred = torch.sigmoid(pred)

#         ious.append(iou_score(pred, mask))
#         dices.append(dice_score(pred, mask))
#         accs.append(accuracy(pred, mask))
#         sens.append(sensitivity(pred, mask))
#         spes.append(specificity(pred, mask))


# print('========== RESULTS =========')

# print(f'IoU  : {sum(ious)/len(ious):.4f}')
# print(f'DSC  : {sum(dices)/len(dices):.4f}')
# print(f'ACC  : {sum(accs)/len(accs):.4f}')
# print(f'SEN  : {sum(sens)/len(sens):.4f}')
# print(f'SPE  : {sum(spes)/len(spes):.4f}')



import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from models.baf_unet import BAFUNet
from utils.metrics import compute_all_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================== CONFIG ======================
IMAGE_SIZE = 512
BATCH_SIZE = 12                    # You can increase this (4~8) if you have enough VRAM
CHECKPOINT_PATH = 'checkpoints/best_model.pth'

# ====================== DATASET ======================
dataset = BasicDataset(
    images_dir='datasets/ISIC2016/test_images',
    mask_dir='datasets/ISIC2016/test_masks',
    image_size=IMAGE_SIZE,
    mask_suffix='_segmentation',
)

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

# ====================== MODEL ======================
model = BAFUNet(n_channels=3, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
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
