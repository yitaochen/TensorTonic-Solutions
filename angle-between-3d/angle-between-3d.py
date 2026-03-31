import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v, w = np.asarray(v), np.asarray(w)
    eps = 1e-10
    if np.linalg.norm(v) < eps or np.linalg.norm(w) < eps:
        return np.nan 

    return np.arccos(np.clip(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)), -1, 1))