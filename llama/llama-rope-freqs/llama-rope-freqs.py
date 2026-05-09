import torch
import math 

def precompute_rope_freqs(max_seq_len, d_head, base=10000.0):
    """
    Returns: tuple of (cos_table, sin_table) both shape (max_seq_len, d_head//2)
    """
    # YOUR CODE HERE
    theta = torch.exp(-torch.arange(0, d_head, 2) / d_head * math.log(base))
    pos = torch.arange(max_seq_len).unsqueeze(1)

    angle = pos * theta

    return torch.cos(angle), torch.sin(angle)