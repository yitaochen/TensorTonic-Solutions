import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    # print(n, p, k)

    if k == 0:
        return ((1-p)**n, (1-p)**n)
    elif k == n:
        return (p**n, 1.0)
    else:
        cdf = 0
        for i in range(k+1):
            pmf = comb(n, i) * p**i * (1-p)**(n-i)
            cdf += pmf
        
        return (float(pmf), float(cdf))
