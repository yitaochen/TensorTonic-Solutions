import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    """
    Add position embeddings to patch embeddings.
    pos_embed: position embedding of shape (1, N, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    N = num_patches + 1
    D = embed_dim
    if pos_embed is None:
        pos_embed = np.random.randn(1, N, D) * 0.02

    return patches + pos_embed