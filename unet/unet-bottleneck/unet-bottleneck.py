import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    Two 3x3 unpadded convolutions, no pooling.
    Returns zero array with correct shape.
    """
    # Your implementation here
    B, H, W, C = x.shape 
    return np.zeros((B, H-4, W-4, out_channels))
