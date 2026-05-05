import torch

def mtp_head(h: torch.Tensor, W_projs: list, W_head: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, num_predict, vocab_size)
    """
    # YOUR CODE HERE
    hd = torch.stack([h @ W_proj.T for W_proj in W_projs], dim=-2)
    logits = hd @ W_head.T

    return logits
    