import torch

def apply_rope(q, k, freqs_cos, freqs_sin):
    """
    Returns: tuple of (q_rotated, k_rotated) same shapes as input
    """
    # YOUR CODE HERE
    q_e = q[:, :, :, 0::2]
    q_o = q[:, :, :, 1::2]
    q_roated = torch.zeros_like(q)
    q_roated[:, :, :, 0::2] = q_e * freqs_cos - q_o * freqs_sin
    q_roated[:, :, :, 1::2] = q_e * freqs_sin + q_o * freqs_cos

    k_e = k[:, :, :, 0::2]
    k_o = k[:, :, :, 1::2]
    k_roated = torch.zeros_like(q)
    k_roated[:, :, :, 0::2] = k_e * freqs_cos - k_o * freqs_sin
    k_roated[:, :, :, 1::2] = k_e * freqs_sin + k_o * freqs_cos

    return (q_roated, k_roated)

    