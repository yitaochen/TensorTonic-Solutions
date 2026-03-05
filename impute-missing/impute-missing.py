import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.asarray(X)
    N = len(X)
    X_copy = X.reshape((N, -1)).copy()
    N, D = X_copy.shape
    for j in range(D):
        mask_nan = np.isnan(X_copy[:, j])
        mask = np.logical_not(np.isnan(X_copy[:, j]))
        if strategy == 'mean' and np.any(mask):
            stat = np.mean(X_copy[mask, j])
        elif strategy == 'median' and np.any(mask):
            stat = np.median(X_copy[mask, j])
        else:
            stat = 0.0
        X_copy[mask_nan, j] = stat 

    return X_copy.reshape(X.shape)