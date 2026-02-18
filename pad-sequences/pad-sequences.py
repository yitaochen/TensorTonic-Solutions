import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if max_len is not None else max(len(seq) for seq in seqs) or 0 
    print(L)
    ans = np.full((N, L), pad_value, dtype=int)
    if not seqs:
        return ans 
    for i, seq in enumerate(seqs):
        l = len(seq)
        ans[i, :min(l, L)] = seq[:min(l, L)]

    return ans 
    
    