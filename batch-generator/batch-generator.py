import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.array(X, copy=False)
    y = np.array(y, copy=False)

    n_samples = X.shape[0]

    indices = np.arange(n_samples)

    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        if drop_last and len(batch_indices) < batch_size:
            break 
        yield X[batch_indices], y[batch_indices]

