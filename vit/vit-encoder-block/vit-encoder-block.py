import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE
    B, N, D = x.shape 
    xhat = layernorm(x)
    Q = (xhat @ Wq).reshape(B, N, num_heads, D//num_heads).swapaxes(1, 2)
    K = (xhat @ Wk).reshape(B, N, num_heads, D//num_heads).swapaxes(1, 2)
    V = (xhat @ Wv).reshape(B, N, num_heads, D//num_heads).swapaxes(1, 2)
    scores = Q @ K.swapaxes(-1, -2) * 1.0 / K.shape[-1]**0.5
    att = softmax(scores, axis=-1) @ V 
    att = att.swapaxes(1, 2).reshape(B, N, D)
    x = x + att @ Wo 
    xhat = layernorm(x)
    hidden_dim = embed_dim * mlp_ratio
    mlp = gelu(xhat @ W1) @ W2 

    return x + mlp 

def softmax(x, axis):
    return np.exp(x - x.max(axis=axis, keepdims=True)) / np.sum(np.exp(x - x.max(axis=axis, keepdims=True)), axis=axis, keepdims=True)
    
def layernorm(x, eps=0):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return (x - mu) / (var + eps)**0.5

def gelu(x):
    return 0.5 * x * (1 + np.tanh((2/np.pi)**0.5 * (x + 0.044715 * x**3)))