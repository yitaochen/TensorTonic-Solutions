import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    gnorm = np.linalg.norm(g.copy())
    # print(gnorm)
    if max_norm <= 0 or gnorm <= max_norm or gnorm == 0:
        return g.copy()
    else:
        return g.copy() * max_norm / gnorm