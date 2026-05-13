import torch

from metrics.sensitivity import sensitivity_score


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

score = sensitivity_score(preds, targets)

print("Sensitivity:", score.item())