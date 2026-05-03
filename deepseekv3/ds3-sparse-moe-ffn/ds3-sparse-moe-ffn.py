import torch

def moe_ffn(x, gate_weight, expert_W_gates, expert_W_ups, expert_W_downs,
           shared_W_gate, shared_W_up, shared_W_down, top_k):
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    scores = torch.softmax(x @ gate_weight.T, dim=-1)
    weights, indices = torch.topk(scores, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    def expert(x, Wg, Wu, Wd):
        return (torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
    shared = expert(x, shared_W_gate, shared_W_up, shared_W_down)

    B, S, D = x.shape
    x_flat = x.reshape(B*S, D)
    indices_flat = indices.reshape(B*S, top_k)
    weights_flat = weights.reshape(B*S, top_k)
    routed_flat = torch.zeros_like(x_flat)
    n_experts = shared_W_gate.shape[0]

    for e in range(n_experts):
        token_idx, slot_idx = torch.where(indices_flat == e)
        if token_idx.numel() == 0:
            continue
        x_e = x_flat[token_idx]
        out_e = expert(x_e, expert_W_gates[e], expert_W_ups[e], expert_W_downs[e])

        w_e = weights_flat[token_idx, slot_idx].unsqueeze(-1) # (n_tokens_for_e, 1) broadcasting to feat dim

        routed_flat.index_add_(0, token_idx, w_e * out_e)

    routed = routed_flat.reshape(B, S, D)

    return shared + routed 

        
    
    

    return shared + routed 

    