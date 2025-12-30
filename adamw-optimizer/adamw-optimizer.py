import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    # print(w, m, v)
    grad = np.asarray(grad)
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad**2
    # print(v.shape == w.shape)
    new_w = w - lr * weight_decay * w - lr * new_m / (np.sqrt(new_v) + eps)
    # print(f"new w, m, v: {w, m, v}")
    return (new_w, new_m, new_v)