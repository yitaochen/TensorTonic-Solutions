import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if len(x) != len(y):
        raise ValueError("Length mismatch of input x and y")
    return x @ y