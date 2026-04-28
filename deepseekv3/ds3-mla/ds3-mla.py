import torch
import math

def mla(x: torch.Tensor, W_dkv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor,
        W_q: torch.Tensor, W_qr: torch.Tensor, W_kr: torch.Tensor, W_o: torch.Tensor,
        cos_freq: torch.Tensor, sin_freq: torch.Tensor, num_heads: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    B, L, _ = x.shape
    c_kv = x @ W_dkv.T
    K_nope = (c_kv @ W_uk.T).view(B, L, num_heads, -1).transpose(1, 2)
    V = (c_kv @ W_uv.T).view(B, L, num_heads, -1).transpose(1, 2)
    Q_nope = (x @ W_q.T).view(B, L, num_heads, -1).transpose(1, 2)
    cos = torch.cat((cos_freq[:L], cos_freq[:L]), dim=-1)[None, None, :, :]
    sin = torch.cat((sin_freq[:L], sin_freq[:L]), dim=-1)[None, None, :, :]
    
    q_rope_input = (x @ W_qr.T).view(B, L, num_heads, -1).transpose(1, 2)
    qx1, qx2 = q_rope_input.chunk(2, dim=-1)
    qrotated = torch.cat((-qx2, qx1), dim=-1)
    q_rope = q_rope_input * cos + qrotated * sin 

    k_rope_input = (x @ W_kr.T).view(B, L, num_heads, -1).transpose(1, 2)
    kx1, kx2 = k_rope_input.chunk(2, dim=-1)
    krotated = torch.cat((-kx2, kx1), dim=-1)
    k_rope = k_rope_input * cos + krotated * sin

    Q = torch.cat((Q_nope, q_rope), dim=-1)
    K = torch.cat((K_nope, k_rope), dim=-1)

    mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
    att = Q @ K.transpose(-1, -2) * 1.0 / math.sqrt(K.size(-1))
    att = att.masked_fill(mask == 0, float("-inf"))
    att = torch.nn.functional.softmax(att, dim=-1)

    out = att @ V 

    out = out.transpose(1, 2).contiguous().view(B, L, -1)

    return out @ W_o.T + x
    