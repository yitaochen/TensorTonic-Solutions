import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)
    prob_real = np.clip(sigmoid(real_data @ D_W), a_min=1e-8, a_max=None)
    prob_fake = sigmoid(fake_data @ D_W)
    d_loss = -np.mean(np.log(prob_real) + np.log((1-prob_fake).clip(min=1e-8, max=None)))
    g_loss = -np.mean(np.log(prob_fake.clip(min=1e-8, max=None)))

    return {"d_loss": d_loss, "g_loss": g_loss}

def sigmoid(x):
    return np.where(x>=0, 1/(1 + np.exp(-x)), np.exp(x)/(np.exp(x) + 1))