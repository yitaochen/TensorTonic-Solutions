import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    unique_values, counts = np.unique(y, return_counts=True)
    non_zero_index = counts != 0
    counts = counts[non_zero_index]
    probs = counts / np.sum(counts)

    return -np.sum(probs*np.log2(probs))