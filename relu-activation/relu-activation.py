import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.atleast_1d(x)
    return np.maximum(x, 0)