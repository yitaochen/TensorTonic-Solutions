import torch

def load_balanced_router(x: torch.Tensor, gate_weight: torch.Tensor, expert_bias: torch.Tensor, top_k: int):
    """
    Returns: tuple (top_k_indices, top_k_weights)
      - top_k_indices: (batch, seq_len, top_k) integer indices of selected experts
      - top_k_weights: (batch, seq_len, top_k) renormalized UNBIASED softmax weights
    """
    # YOUR CODE HERE
    scores = torch.softmax(x @ gate_weight.T, dim=-1)
    biased_scores = scores + expert_bias
    values, top_k_indices = torch.topk(biased_scores, top_k)
    top_k_weights = torch.gather(scores, -1, top_k_indices)

    return top_k_indices, top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)