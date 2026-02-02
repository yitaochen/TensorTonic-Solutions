import numpy as np 

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)

    return (X @ W + b).tolist()