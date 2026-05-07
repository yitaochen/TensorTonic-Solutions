import torch
import math

def ds3_block(x, W_q, W_k, W_v, W_o, gamma_attn, gamma_ffn,
              W_gate_ffn, W_up_ffn, W_down_ffn,
              gate_weight, expert_W_gates, expert_W_ups, expert_W_downs,
              shared_W_gate, shared_W_up, shared_W_down,
              layer_idx, num_dense_layers, top_k, num_heads, eps=1e-6):
    """
    Single DeepSeek V3 transformer block.
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    B, L, D = x.shape
    rms_attn = (torch.mean(x**2, dim=-1, keepdim=True) + eps) ** 0.5
    xhat = x * gamma_attn / rms_attn
    Q = (xhat @ W_q.T).view(B, L, num_heads, D // num_heads).transpose(1, 2)
    K = (xhat @ W_k.T).view(B, L, num_heads, D // num_heads).transpose(1, 2)
    V = (xhat @ W_v.T).view(B, L, num_heads, D // num_heads).transpose(1, 2)

    mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
    att = Q @ K.transpose(-1, -2) * 1.0 / math.sqrt(K.size(-1))
    att = att.masked_fill(mask == 0, float("-inf"))
    att = torch.nn.functional.softmax(att, dim=-1)

    out = att @ V 

    out = out.transpose(1, 2).contiguous().view(B, L, -1)

    x1 = x + out @ W_o.T 

    rms_ffn = (torch.mean(x1**2, dim=-1, keepdim=True) + eps) ** 0.5
    x1hat = x1 * gamma_ffn / rms_ffn

    def expert(x, Wg, Wu, Wd):
        return (torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
    if layer_idx < num_dense_layers:
        return x1 + expert(x1hat, W_gate_ffn, W_up_ffn, W_down_ffn)
    else:
        scores = torch.softmax(x1hat @ gate_weight.T, dim=-1)
        weights, indices = torch.topk(scores, top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        shared = expert(x1hat, shared_W_gate, shared_W_up, shared_W_down)

        x1hat_flat = x1hat.reshape(B*L, D)
        indices_flat = indices.reshape(B*L, top_k)
        weights_flat = weights.reshape(B*L, top_k)
        
        routed_flat = torch.zeros_like(x1hat_flat)
        
        n_experts = shared_W_gate.shape[0]

        for e in range(n_experts):
            token_idx, slot_idx = torch.where(indices_flat == e)
            if token_idx.numel() == 0:
                continue 
            x1hat_flat_e = x1hat_flat[token_idx]
            out_e = expert(x1hat_flat_e, expert_W_gates[e], expert_W_ups[e], expert_W_downs[e])
            
            routed_flat.index_add_(0, token_idx, out_e * weights_flat[token_idx, slot_idx].unsqueeze(1))

        routed = routed_flat.reshape(B, L, D)

        return x1 + shared + routed