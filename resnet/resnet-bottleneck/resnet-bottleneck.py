import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    x, W1, W2, W3, Ws = map(np.asarray, [x, W1, W2, W3, Ws])

    return relu(relu(relu(x @ W1) @ W2) @ W3 + x @ Ws)

def relu(x):
    return np.where(x>=0, x, 0)
