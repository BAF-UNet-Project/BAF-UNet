import torch

from metrics.iou import iou_score


preds = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])

targets = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
])

score = iou_score(preds, targets)

print("IoU Score:", score.item())