import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng:
        arr_rng = rng.random(size=x.shape)
    else:
        arr_rng = np.random.random(size=x.shape)
    output_pattern = np.where(arr_rng < 1- p, 1/(1-p), 0)
    return x*output_pattern, output_pattern