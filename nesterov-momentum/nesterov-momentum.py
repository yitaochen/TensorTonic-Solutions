import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w = np.asarray(w)
    v = np.asarray(v)
    grad = np.asarray(grad)
    wlook = w - momentum * v
    new_v = momentum * v + lr * grad
    new_w = w - new_v

    return (new_w, new_v)