import torch
import torch.nn.functional as F
import math

def llama_forward(token_ids, W_embed, blocks, rms_final, W_head, freqs_cos, freqs_sin, eps=1e-6):
    """
    Returns: logits tensor (batch, seq_len, vocab_size) from Llama 3 forward pass.
    """
    h = W_embed[token_ids]
    B, S, D = h.shape
    for i, block in enumerate(blocks):
        rms_w1 = block["rms_w1"]
        rms_w2 = block["rms_w2"]
        W_q = block["W_q"]
        W_k = block["W_k"]
        W_v = block["W_v"]
        W_o = block["W_o"]
        W_gate = block["W_gate"]
        W_up = block["W_up"]
        W_down = block["W_down"]
        n_heads = block["n_heads"]
        n_kv_heads = block["n_kv_heads"]
        hhat = rms_norm(h, rms_w1, eps)
        Q = (hhat @ W_q.T).view(B, S, n_heads, D//n_heads).transpose(1, 2)
        K = (hhat @ W_k.T).view(B, S, n_kv_heads, D//n_heads).transpose(1, 2)
        V = (hhat @ W_v.T).view(B, S, n_kv_heads, D//n_heads).transpose(1, 2)
        Q = apply_rope(Q, freqs_cos[:S], freqs_sin[:S])
        K = apply_rope(K, freqs_cos[:S], freqs_sin[:S])
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K[:, :, None, :, :].expand(B, n_kv_heads, n_rep, S, D//n_heads).reshape(B, n_heads, S, D//n_heads)
            V = V[:, :, None, :, :].expand(B, n_kv_heads, n_rep, S, D//n_heads).reshape(B, n_heads, S, D//n_heads)
        mask = torch.tril(torch.ones(S, S)).view(1, 1, S, S)
        att = Q @ K.transpose(-1, -2) * 1.0 / math.sqrt(K.size(-1))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ V 
        out = out.transpose(1, 2).contiguous().reshape(B, S, D)
        out = out @ W_o.T 
    
        h = h + out 
        hhat = rms_norm(h, rms_w2, eps)
        ffn = swiglu_ffn(hhat, W_gate, W_up, W_down)
    
        h = h + ffn

    h = rms_norm(h, rms_final, eps)
    logits = h @ W_head.T

    return logits

def rms_norm(x, w, eps):
    return x * w / (torch.mean(x**2, dim=-1, keepdim=True) + eps) ** 0.5

def apply_rope(t, freqs_cos, freqs_sin):
    freqs_cos = freqs_cos[None, None, :, :]
    freqs_sin = freqs_sin[None, None, :, :]
    t_even = t[..., 0::2]
    t_odd = t[..., 1::2]
    t_rotated = torch.zeros_like(t)
    t_rotated[..., 0::2] = t_even * freqs_cos - t_odd * freqs_sin
    t_rotated[..., 1::2] = t_even * freqs_sin + t_odd * freqs_cos
    return t_rotated

def swiglu_ffn(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T