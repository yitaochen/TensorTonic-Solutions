import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    return np.round(np.linspace(beta_1, beta_T, T), 6).tolist()

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    series = (np.cos((np.arange(T+1)/T + s)/(1 + s) * np.pi/2))**2
    return np.clip(np.round(series[1:] / series[0], 6), 0.0001, 0.9999).tolist()

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    T = len(alpha_bars)
    betas = np.zeros(T)
    betas[0] = 1 - alpha_bars[0]
    for i in range(1, T):
        betas[i] = 1 - alpha_bars[i]/alpha_bars[i-1]

    return np.clip(np.round(betas, 6), 0.0001, 0.9999).tolist()