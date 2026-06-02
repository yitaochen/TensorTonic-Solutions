import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    h = relu(x @ W1.T)
    return relu(h @ W2.T + x)


def relu(x):
    return np.where(x > 0, x, 0)
