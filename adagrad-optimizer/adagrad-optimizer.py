import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)
    new_G = G + g**2
    new_w = w - lr * g / (np.sqrt(new_G) + eps)
    # print(w, g, G)
    return (new_w, new_G)