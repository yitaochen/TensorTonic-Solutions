import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    n, d = X_centered.shape
    # print(f"n: {n} d: {d}")
    C = X_centered.T @ X_centered / (n - 1)

    W = []

    for i in range(k):
        v = np.random.rand(d)
        v = v / np.linalg.norm(v)
        for _ in range(100):
            v_new = C @ v 
            v = v_new / np.linalg.norm(v_new)
        W.append(v)
        eigen_val = (v.T @ C) @ v
        C -= eigen_val * v[:, None] * v[None, :]

    W = np.hstack(W).reshape((d, k))
    # print(W.shape)
    ans = (X_centered @ W)

    return ans.tolist()

