import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    state = {'min': np.full(shape=D, fill_value=float("inf")), 'max': np.full(shape=D, fill_value=float("-inf"))}
    return state

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    state['min'] = np.minimum(state['min'], np.min(X_batch, axis=0))
    state['max'] = np.maximum(state['max'], np.max(X_batch, axis=0))
    return (X_batch - state['min']) / (state['max'] - state['min'] + eps)