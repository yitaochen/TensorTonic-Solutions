import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE
    h_cls = encoder_output[:, 0, :]
    B, D = h_cls.shape
    hhat = (h_cls - h_cls.mean(axis=-1, keepdims=True)) / h_cls.std(axis=-1, keepdims=True)
    if W_head is None:
        W_head = np.random.randn(D, num_classes) * 0.02
    return hhat @ W_head