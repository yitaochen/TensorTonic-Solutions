import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        A = np.asarray(matrix)
    except:
        return None 
    if A.ndim != 2:
        return None 
    if A.shape[0] != A.shape[1]:
        return None 
    eigvals = np.linalg.eigvals(A)
    np.lexsort(eigvals)

    return eigvals