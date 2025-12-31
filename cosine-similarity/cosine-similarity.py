import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if not norma or not normb:
        return 0.0
    
    return a @ b / (norma * normb)