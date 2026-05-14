import torch
import torch.nn.functional as F
import math

def llama_block(x, rms_w1, rms_w2, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, W_gate, W_up, W_down, freqs_cos, freqs_sin, eps=1e-6):
    """
    Returns: dict with key "output" containing tensor (batch, seq_len, d_model) as nested list, rounded to 4 decimals.
    """
    # YOUR CODE HERE
    x = torch.tensor(x)
    rms_w1 = torch.tensor(rms_w1)
    rms_w2 = torch.tensor(rms_w2)
    W_q = torch.tensor(W_q)
    W_k = torch.tensor(W_k)
    W_v = torch.tensor(W_v)
    W_o = torch.tensor(W_o)
    W_gate = torch.tensor(W_gate)
    W_up = torch.tensor(W_up)
    W_down = torch.tensor(W_down)
    freqs_cos = torch.tensor(freqs_cos)
    freqs_sin = torch.tensor(freqs_sin)
    B, S, D = x.shape
    xhat = rms_norm(x, rms_w1, eps)
    Q = (xhat @ W_q.T).view(B, S, n_heads, D//n_heads).transpose(1, 2)
    K = (xhat @ W_k.T).view(B, S, n_kv_heads, D//n_heads).transpose(1, 2)
    V = (xhat @ W_v.T).view(B, S, n_kv_heads, D//n_heads).transpose(1, 2)
    Q = apply_rope(Q, freqs_cos, freqs_sin)
    K = apply_rope(K, freqs_cos, freqs_sin)
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        K = K[:, :, None, :, :].expand(B, n_kv_heads, n_rep, S, D//n_heads).reshape(B, n_heads, S, D//n_heads)
        V = V[:, :, None, :, :].expand(B, n_kv_heads, n_rep, S, D//n_heads).reshape(B, n_heads, S, D//n_heads)
    att = torch.softmax(Q @ K.transpose(-1, -2) * 1.0 / math.sqrt(K.size(-1)), dim=-1)
    out = att @ V 
    out = out.transpose(1, 2).contiguous().reshape(B, S, D)
    out = out @ W_o.T 

    h = x + out 
    hhat = rms_norm(h, rms_w2, eps)
    ffn = swiglu_ffn(hhat, W_gate, W_up, W_down)

    output = h + ffn 
    
    return {"output": torch.round(output, decimals=4)}
        

def rms_norm(x, w, eps):
    return x * w / (torch.mean(x**2, dim=-1, keepdim=True) + eps) ** 0.5

def apply_rope(t, freqs_cos, freqs_sin):
    t_even = t[..., 0::2]
    t_odd = t[..., 1::2]
    t_rotated = torch.zeros_like(t)
    t_rotated[..., 0::2] = t_even * freqs_cos - t_odd * freqs_sin
    t_rotated[..., 1::2] = t_even * freqs_sin + t_odd * freqs_cos
    return t_rotated

def swiglu_ffn(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T