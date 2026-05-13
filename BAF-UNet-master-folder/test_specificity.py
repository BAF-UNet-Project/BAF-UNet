import torch
from metrics.specificity import specificity

preds = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
])

targets = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])

score = specificity(preds, targets)

print("Specificity:", score.item())