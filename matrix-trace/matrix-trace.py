import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input should be a square matrix!")
    tr = 0 
    for i in range(A.shape[0]):
        tr += A[i, i]
    
    return tr 
