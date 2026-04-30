import torch

def shared_expert_ffn(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    gate = swish(x @ W_gate.T)
    up = x @ W_up.T
    output = (gate * up) @ W_down.T

    return output 
        