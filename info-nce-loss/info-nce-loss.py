import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    S = Z1 @ Z2.T / temperature
    S = np.exp(S - np.max(S, axis=1, keepdims=True))
    N = len(S)
    L = -1/N * np.sum(np.log(np.diag(S)/np.sum(S, axis=1)))
    return L 