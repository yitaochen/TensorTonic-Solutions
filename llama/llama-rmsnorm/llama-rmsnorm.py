import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: Normalized tensor of same shape as x
    """
    # YOUR CODE HERE
    return x * gamma / ((x**2).mean(dim=-1, keepdim=True) + eps)**0.5