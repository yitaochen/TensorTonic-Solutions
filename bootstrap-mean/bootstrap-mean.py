import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x)
    N = len(x)
    if rng is not None:
        indices = rng.integers(low=0, high=N, size=(n_bootstrap, N)) 
    else:
        indices = np.random.randint(low=0, high=N, size=(n_bootstrap, N)) 
    # print(indices.shape)
    bootstrap_samples = x[indices]
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    alpha = 1 - ci 
    lower = np.quantile(bootstrap_means, alpha/2)
    upper = np.quantile(bootstrap_means, 1- alpha/2)
    return (bootstrap_means, lower, upper)
