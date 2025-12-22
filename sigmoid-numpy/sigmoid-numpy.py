import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    shape = x.shape
    # temp = 1 / (1 + np.exp(-x.reshape((-1, 1))))
    # print(temp)
    return (1 / (1 + np.exp(-x.reshape((-1, 1))))).reshape(shape)