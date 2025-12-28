import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x)
    vectorized_erf = np.vectorize(math.erf)
    return 0.5 * x * (1 + vectorized_erf(x/np.sqrt(2)))