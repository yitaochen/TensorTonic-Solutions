import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores_copy = np.asarray(scores).copy()
    T = scores_copy.shape[-1]
    mask = np.triu(np.full((T, T), True, dtype=bool), k=1)
    scores_copy[..., mask] = mask_value

    return scores_copy