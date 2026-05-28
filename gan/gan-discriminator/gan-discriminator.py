import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x = np.asarray(x)
    W = np.asarray(W)
    return np.round(sigmoid(x @ W), 4)

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1))