import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    cdf = 0
    for i in range(k+1):
        pmf = np.exp(-lam) * lam**i / np.exp(np.sum(np.log(np.arange(1, i+1))))
        cdf += pmf
    # print(cdf)
    return (float(pmf), float(cdf))