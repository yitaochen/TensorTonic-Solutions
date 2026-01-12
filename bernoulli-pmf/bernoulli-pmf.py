import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x)
    pmf = np.where(x==1, p, 1-p)
    # print(x, p)
    return (pmf, float(p), float(p*(1-p)))