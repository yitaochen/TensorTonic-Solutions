import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, N, D = patches.shape
    if cls_token is None:
        cls_token = np.random.randn(1, 1, D) * 0.02 
    cls_token = np.repeat(cls_token, B, axis=0)
    return np.concatenate((cls_token, patches), axis=1)
        
    