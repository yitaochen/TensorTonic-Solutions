import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    mse = np.mean(np.sum((x-x_recon)**2, axis=1))
    kl = np.mean(np.sum(-0.5 * (1-np.exp(log_var)-mu**2+log_var), axis=1))

    return {"total": mse+kl, "recon": mse, "kl": kl}
