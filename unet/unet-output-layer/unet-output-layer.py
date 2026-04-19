import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    Preserves spatial dims, changes channels to num_classes.
    Returns zero array with correct shape.
    """
    # Your implementation here
    B, H, W, C = features.shape
    return np.zeros((B, H, W, num_classes))
