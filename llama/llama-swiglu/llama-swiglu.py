import torch

def swiglu_ffn(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """
    Apply SwiGLU feed-forward network.
    """
    # Your code here
    return (torch.nn.functional.silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T