import torch

from metrics.dice import dice_score


# Fake prediction
preds = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Ground truth
targets = torch.tensor([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
])

score = dice_score(preds, targets)

print("Dice Score:", score.item())