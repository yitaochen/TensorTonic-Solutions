import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    B, H, W, C = x.shape
    # 4 encoders
    for i in range(4):
        H = (H-4)//2
        W = (W-4)//2
    # bottlenect
    H -= 4
    W -= 4
    # 4 decoders
    for i in range(4):
        H = 2*H - 4
        W = 2*W - 4
    

    return np.zeros((B, H, W, num_classes))
