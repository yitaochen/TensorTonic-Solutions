import torch
import math

def ds3_forward(token_ids, W_embed, blocks_params, gamma_final, W_lm_head,
                num_dense_layers, top_k, num_heads, eps=1e-6):
    """
    Full DeepSeek V3 forward pass.
    Returns: torch.Tensor of shape (batch, seq_len, vocab_size)
    """
    # YOUR CODE HERE
    h = W_embed[token_ids]
    B, L, D = h.shape
    
    def expert(x, Wg, Wu, Wd):
        return (torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
        
    for i, blocks_params_i in enumerate(blocks_params):
        W_q = blocks_params_i["W_q"]
        W_k = blocks_params_i["W_k"]
        W_v = blocks_params_i["W_v"]
        W_o = blocks_params_i["W_o"]
        gamma_att = blocks_params_i["gamma_attn"]
        h_rms = h * gamma_att / (torch.mean(h**2, dim=-1, keepdim=True) + eps)**0.5
        Q = (h_rms @ W_q.T).view(B, L, num_heads, D//num_heads).transpose(1, 2)
        K = (h_rms @ W_k.T).view(B, L, num_heads, D//num_heads).transpose(1, 2)
        V = (h_rms @ W_v.T).view(B, L, num_heads, D//num_heads).transpose(1, 2)
        mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
        att = Q @ K.transpose(-1, -2) * 1.0 / math.sqrt(K.size(-1))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        out = att @ V 
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        h = h + out @ W_o.T
        gamma_ffn = blocks_params_i["gamma_ffn"]
        h_rms = h * gamma_ffn / (torch.mean(h**2, dim=-1, keepdim=True) + eps)**0.5
        if i < num_dense_layers:
            W_gate_ffn = blocks_params_i["W_gate_ffn"]
            W_up_ffn = blocks_params_i["W_up_ffn"]
            W_down_ffn = blocks_params_i["W_down_ffn"]
            h = h + expert(h_rms, W_gate_ffn, W_up_ffn, W_down_ffn)
        else:
            shared_W_gate = blocks_params_i["shared_W_gate"]
            shared_W_up = blocks_params_i["shared_W_up"]
            shared_W_down = blocks_params_i["shared_W_down"]
            expert_W_gates = blocks_params_i["expert_W_gates"]
            expert_W_ups = blocks_params_i["expert_W_ups"]
            expert_W_downs = blocks_params_i["expert_W_downs"]
            gate_weight = blocks_params_i["gate_weight"]
            scores = torch.softmax(h_rms @ gate_weight.T, dim=-1) 
            weights, indices = torch.topk(scores, top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)

            shared = expert(h_rms, shared_W_gate, shared_W_up, shared_W_down)

            h_rms_flat = h_rms.reshape(B*L, D)
            indices = indices.reshape(B*L, top_k)
            weights = weights.reshape(B*L, top_k)
            routed_flat = torch.zeros_like(h_rms_flat)
            
            n_experts = len(expert_W_gates)

            for e in range(n_experts):
                token_idx, slot_idx = torch.where(indices == e)
                h_rms_flat_e = h_rms_flat[token_idx]
                out_e = expert(h_rms_flat_e, expert_W_gates[e], expert_W_ups[e], expert_W_downs[e])
                w_e = weights[token_idx, slot_idx].unsqueeze(1)
                routed_flat.index_add_(0, token_idx, out_e * w_e)

            routed = routed_flat.reshape(B, L, D)

            h = h + shared + routed

    h_rms = h * gamma_final / (torch.mean(h**2, dim=-1, keepdim=True) + eps)**0.5

    logits = h_rms @ W_lm_head.T

    return logits
            
            
            