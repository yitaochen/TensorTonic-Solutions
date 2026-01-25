import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    indices = np.arange(N)
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)
    chunks = np.array_split(indices, k)
    ans = []
    for i in range(k):
        val_idx = chunks[i]
        train_idx = np.concatenate(chunks[:i] + chunks[i+1:])
        ans.append((train_idx, val_idx))
    
    return ans 
