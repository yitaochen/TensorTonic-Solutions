import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    N = len(X)
    if N < 2:
        return None
    X = np.asarray(X)
    if X.ndim < 2:
        return None 
    mu = np.mean(X, axis=0)
    sigma = (X - mu).T @ (X - mu) / (N - 1) 

    # print(sigma)
    return sigma