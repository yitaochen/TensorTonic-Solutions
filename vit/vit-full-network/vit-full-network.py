import numpy as np

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        # Initialize weights here
        if W_patch is None:
            self.W_patch = np.random.randn(patch_size*patch_size*3, embed_dim) * 0.02
        else:
            self.W_patch = W_patch
        if cls_token is None:
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        else:
            self.cls_token = cls_token
        if pos_embed is None:
            self.pos_embed = np.random.randn(1, self.num_patches+1, embed_dim) * 0.02
        else:
            self.pos_embed = pos_embed
        hidden_dim = embed_dim * mlp_ratio
        if encoder_weights is None:
            self.encoder_weights = []
            for d in range(depth):
                mp = {}
                mp['W1'] = np.random.randn(embed_dim, hidden_dim) * 0.02
                mp['W2'] = np.random.randn(hidden_dim, embed_dim) * 0.02
                mp['Wq'] = np.random.randn(embed_dim, embed_dim) * 0.02
                mp['Wk'] = np.random.randn(embed_dim, embed_dim) * 0.02
                mp['Wv'] = np.random.randn(embed_dim, embed_dim) * 0.02
                mp['Wo'] = np.random.randn(embed_dim, embed_dim) * 0.02
                self.encoder_weights.append(mp)
        else:
            self.encoder_weights = encoder_weights
        if W_head is None:
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02
        else:
            self.W_head = W_head

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        B, H, W, C = x.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        patch_dim = self.patch_size * self.patch_size * C 
        patches = x.reshape(B, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, C).swapaxes(2, 3).reshape(B, N, patch_dim)
        z = patches @ self.W_patch
        cls_token = np.repeat(self.cls_token, B, axis=0)
        z = np.concatenate((cls_token, z), axis=1)
        N += 1
        z += self.pos_embed
        for encoder_weights in self.encoder_weights:
            W1 = encoder_weights['W1']
            W2 = encoder_weights['W2']
            Wq = encoder_weights['Wq']
            Wk = encoder_weights['Wk']
            Wv = encoder_weights['Wv']
            Wo = encoder_weights['Wo']
            zhat = self.layernorm(z)
            Q = (zhat @ Wq).reshape(B, N, self.num_heads, self.embed_dim//self.num_heads).swapaxes(1, 2)
            K = (zhat @ Wk).reshape(B, N, self.num_heads, self.embed_dim//self.num_heads).swapaxes(1, 2)
            V = (zhat @ Wv).reshape(B, N, self.num_heads, self.embed_dim//self.num_heads).swapaxes(1, 2)
            scores = Q @ K.swapaxes(-1, -2) * 1.0 / K.shape[-1]**0.5
            att = self.softmax(scores, axis=-1) @ V 
            att = att.swapaxes(1, 2).reshape(B, N, self.embed_dim)
            z += att @ Wo 
            zhat = self.layernorm(z)
            mlp = self.gelu(zhat @ W1) @ W2
            z += mlp 

        logits = self.layernorm(z[:, 0, :]) @ self.W_head 

        return logits

    def softmax(self, x, axis):
        return np.exp(x - x.max(axis=axis, keepdims=True)) / np.sum(np.exp(x - x.max(axis=axis, keepdims=True)), axis=axis, keepdims=True)
    
    def layernorm(self, x, eps=1e-6):
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
    
        return (x - mu) / (var + eps)**0.5
    
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh((2/np.pi)**0.5 * (x + 0.044715 * x**3)))