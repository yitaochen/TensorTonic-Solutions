import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    intersection = np.sum(p * y)
    return 1 - (2*intersection + eps) / (np.sum(p) + np.sum(y) + eps)