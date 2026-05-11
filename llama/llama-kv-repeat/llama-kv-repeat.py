import torch

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Returns: (batch, n_kv_heads * n_rep, seq_len, d_head)
    """
    # YOUR CODE HERE
    B, n_kv_heads, S, D = kv.shape
    return kv[:, :, None, :, :].expand(B, n_kv_heads, n_rep, S, D).reshape(B, n_kv_heads*n_rep, S, D)