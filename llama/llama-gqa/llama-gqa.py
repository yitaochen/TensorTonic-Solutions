import torch
import torch.nn.functional as F
import math

def grouped_query_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor,
                            W_v: torch.Tensor, W_o: torch.Tensor,
                            n_heads: int, n_kv_heads: int) -> torch.Tensor:
    """
    Returns: (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    B, L, D = x.shape
    Q = (x @ W_q.T).view(B, L, n_heads, D//n_heads).transpose(1, 2)
    K = (x @ W_k.T).view(B, L, n_kv_heads, D//n_heads).transpose(1, 2)
    V = (x @ W_v.T).view(B, L, n_kv_heads, D//n_heads).transpose(1, 2)

    K = K[:, :, None, :, :].expand(B, n_kv_heads, n_heads//n_kv_heads, L, D//n_heads).reshape(B, n_heads, L, D//n_heads)
    V = V[:, :, None, :, :].expand(B, n_kv_heads, n_heads//n_kv_heads, L, D//n_heads).reshape(B, n_heads, L, D//n_heads)

    att = F.softmax(Q @ K.transpose(-2, -1) * 1.0 / math.sqrt(K.size(-1)), dim=-1)
    out = att @ V 

    out = out.transpose(1, 2).contiguous().reshape(B, L, D)

    return out @ W_o.T 