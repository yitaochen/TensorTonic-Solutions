import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model), dtype=float)
    positions = np.arange(seq_len, dtype=float)[:, None]
    div_terms = np.exp(np.arange(0, d_model, 2, dtype=float)*(-np.log(base)/d_model))
    pe[:, 0::2] = np.sin(positions * div_terms)
    pe[:, 1::2] = np.cos(positions * div_terms[:pe[:, 1::2].shape[1]]) # need to handle odd d_model edge case
    
    return pe

def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    x = np.asarray(x, dtype=float)
    B, T, d_model = x.shape
    pe = positional_encoding(T, d_model, base)
    return x + pe[None, :T, :]
    