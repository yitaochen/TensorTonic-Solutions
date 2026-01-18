import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)
    mean = np.mean(x)
    median = np.median(x)
    ctr = Counter(x)
    max_f = max(v for k, v in ctr.items())
    mode = min(k for k, v in ctr.items() if v == max_f)

    return (float(mean), float(median), float(mode))