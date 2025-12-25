import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        mu = np.mean(x, axis=0, keepdims=True)
        var = np.mean((x-mu)**2, axis=0, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + eps)
        y = gamma[None, :] *  xhat + beta[None, :]
    elif x.ndim == 4:
        mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.mean((x-mu)**2, axis=(0, 2, 3), keepdims=True)
        xhat = (x - mu) / np.sqrt(var + eps)
        y = gamma[None, :, None, None] *  xhat + beta[None, :, None, None]
    
    return y

