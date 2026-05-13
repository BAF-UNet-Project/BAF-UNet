import torch


def accuracy_score(preds, targets):
    """
    Computes pixel accuracy.

    Args:
        preds: predicted masks
        targets: ground truth masks

    Returns:
        accuracy score
    """

    preds = preds.float()
    targets = targets.float()

    correct = (preds == targets).float().sum()

    total = torch.numel(preds)

    accuracy = correct / total

    return accuracy