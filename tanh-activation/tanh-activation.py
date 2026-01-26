import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.atleast_1d(x)
    # print(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))