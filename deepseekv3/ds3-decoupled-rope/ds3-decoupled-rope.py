import torch

def decoupled_rope(k_nope: torch.Tensor, k_rope_input: torch.Tensor,
                   cos_freq: torch.Tensor, sin_freq: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, heads, seq, d_nope + d_rope)
    """
    # YOUR CODE HERE
    B, H, L, D = k_nope.shape
    k_rope_input = k_rope_input.reshape(B, L, H, -1).transpose(1, 2)
    x1, x2 = k_rope_input.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)

    cos = torch.cat((cos_freq[:L], cos_freq[:L]), dim=-1)[None, None, :, :]
    sin = torch.cat((sin_freq[:L], sin_freq[:L]), dim=-1)[None, None, :, :]
    k_rope = cos * k_rope_input + sin * rotated

    return torch.cat((k_nope, k_rope), dim=-1)
    