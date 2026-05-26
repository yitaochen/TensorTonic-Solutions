import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Returns: np.ndarray of the final denoised sample
    """
    # YOUR CODE HERE
    T = len(betas)
    betas = np.asarray(betas)
    epsilon_preds = np.asarray(epsilon_preds)
    z_values = np.asarray(z_values)
    alphas = 1 - betas 
    alpha_bars = np.cumprod(alphas)
    x_t = np.asarray(x_T)
    for i in range(T):
        t = T - 1 - i
        alpha_t = alphas[t]
        alpha_bars_t = alpha_bars[t]
        z_t = z_values[i] if i < T - 1 else 0 
        x_t = 1.0 / alpha_t ** 0.5 *(x_t - betas[t]/(1 - alpha_bars_t)**0.5 * epsilon_preds[i]) + betas[t]**0.5 * z_t 

    return x_t 