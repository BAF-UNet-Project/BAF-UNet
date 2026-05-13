import torch

from metrics.accuracy import accuracy_score


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

score = accuracy_score(preds, targets)

print("Accuracy:", score.item())