import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    betas = np.asarray(betas)
    return np.round(np.cumprod(1 - betas), 6)

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE
    x_0 = np.asarray(x_0)
    betas = np.asarray(betas)
    alpha_bar = get_alpha_bar(betas)
    x_shape = x_0.shape
    if epsilon is None:
        epsilon = np.random.randn(*x_shape)
    else:
        epsilon = np.asarray(epsilon)
    
    x_t = np.sqrt(alpha_bar[t-1]) * x_0 + np.sqrt(1 - alpha_bar[t-1]) * epsilon 

    return x_t