import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, H, W, C = image.shape
    N = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C 
    patches = image.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C).swapaxes(2, 3).reshape(B, N, patch_dim)
    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    return patches @ W_proj
    