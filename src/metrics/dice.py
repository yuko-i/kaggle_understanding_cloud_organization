import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = 0.5,
):

    #outputs = nn.Sigmoid()(outputs)
    outputs = outputs.cpu()
    targets = targets.cpu()

    outputs = F.softmax(outputs, dim=0)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice
