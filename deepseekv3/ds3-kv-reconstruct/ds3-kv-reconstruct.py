import torch

def kv_reconstruct(c_kv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor, num_heads: int):
    """
    Returns: tuple (K, V) with K shape (batch, heads, seq, d_nope) and V shape (batch, heads, seq, d_head)
    """
    # YOUR CODE HERE
    B, L, _ = c_kv.shape
    K = (c_kv @ W_uk.T).view(B, L, num_heads, -1).transpose(1, 2)
    V = (c_kv @ W_uv.T).view(B, L, num_heads, -1).transpose(1, 2)

    return (K, V)