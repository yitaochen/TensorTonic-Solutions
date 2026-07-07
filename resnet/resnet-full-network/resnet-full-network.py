import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):

    x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc = map(np.asarray, [x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc])
    relu = lambda z: np.maximum(0, z)

    # Initial conv (linear proj 13 -> 64) + ReLU
    out = relu(x @ conv1)

    # Block 1 — identity shortcut (dims unchanged, 64 -> 64)
    identity = out
    f = relu(out @ W1_b1)     # ReLU(W1 x)
    f = f @ W2_b1             # W2 ReLU(W1 x) = F(x)
    out = relu(f + identity)  # ReLU(F(x) + x)

    # Block 2 — projection shortcut (dim change, 64 -> 128)
    shortcut = out @ Ws_b2    # W_proj x
    f = relu(out @ W1_b2)
    f = f @ W2_b2
    out = relu(f + shortcut)  # ReLU(F(x) + W_proj x)

    # Classification head
    return out @ fc