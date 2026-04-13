import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    N, C_in, H, Wi = x.shape
    C_out, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = Wi - KW + 1

    y = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    y[n, co, i, j] = np.sum(x[n, :, i:i+KH, j:j+KW] * W[co]) + b[co]

    return y