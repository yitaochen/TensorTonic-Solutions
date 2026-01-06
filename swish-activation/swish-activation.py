import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.atleast_1d(x)
    sigma = np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(np.exp(x)+1))
    # print(x.shape)
    return x * sigma