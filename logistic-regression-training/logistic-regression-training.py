import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    N = X.shape[0]
    y = np.array(y)
    w = np.zeros((X.shape[1], ), dtype=np.float32)
    b = 0.0
    for step in range(steps):
        p = _sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / N
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    
    return (w, b)
