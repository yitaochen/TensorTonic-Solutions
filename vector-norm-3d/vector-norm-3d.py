import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v)
    if v.ndim == 1:
        return np.sqrt(np.sum(v**2))
    else:
        return np.sqrt(np.sum(v**2, axis=1))