import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    x_t = np.asarray(x_t, dtype=np.float64)
    epsilon_pred = np.asarray(epsilon_pred, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    alpha_t = (1 - betas)[t-1]
    alpha_bar_t = np.cumprod(1 - betas)[t-1]
    if t == 1:
        z = 0
    else:
        if z is None:
            z = np.random.randn(*x_t.shape)
        else:
            z = np.asarray(z, dtype=np.float64)

    return 1/alpha_t**0.5 * (x_t - betas[t-1]/(1-alpha_bar_t)**0.5 * epsilon_pred) + betas[t-1]**0.5 * z
    