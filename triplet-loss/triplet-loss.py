import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    if anchor.ndim < 2:
        anchor = anchor.reshape((1, -1))
        positive = positive.reshape((1, -1))
        negative = negative.reshape((1, -1))
    d_ap = np.sum((anchor - positive)**2, axis=1)
    d_an = np.sum((anchor - negative)**2, axis=1)
    L = np.maximum(0, d_ap - d_an + margin)
    return np.mean(L)