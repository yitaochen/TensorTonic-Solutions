import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    N = len(y)
    unique_vals, counts = np.unique(y, return_counts=True)
    if num_classes is None:
        num_classes = unique_vals[-1] + 1
    indices = np.searchsorted(np.arange(num_classes), y)
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), indices] = 1

    return one_hot