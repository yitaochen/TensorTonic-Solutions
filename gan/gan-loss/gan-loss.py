import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    real_probs = np.clip(np.asarray(real_probs), a_min=1e-8, a_max=None)
    fake_probs = np.clip(np.asarray(fake_probs), a_min=None, a_max=1-1e-8)
    return np.round(-np.mean(np.log(real_probs) + np.log(1 - fake_probs)), 4)

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    fake_probs = np.clip(np.asarray(fake_probs), a_min=1e-8, a_max=None)
    return np.round(-np.mean(np.log(fake_probs)), 4)