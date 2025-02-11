import torch.nn as nn

def l1_loss(pred, gt, subset, smooth):
    loss = 0.0
    for mod in subset:
        if smooth:
            loss+=nn.SmoothL1Loss(reduction='mean')(pred[mod], gt[mod]) / 2
        else:
            loss+=nn.L1Loss(reduction='mean')(pred[mod], gt[mod])  / 2
    return loss / len(subset)
