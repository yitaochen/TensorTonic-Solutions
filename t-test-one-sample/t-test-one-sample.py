import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    xbar = np.mean(x)
    n = len(x)
    std = np.sqrt(np.sum((x-xbar)**2)/(n-1))

    return (xbar - mu0) * np.sqrt(n) / std 