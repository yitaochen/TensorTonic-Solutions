import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    pre_act = np.asarray(x_t) @ np.asarray(Wx) + np.asarray(h_prev) @ np.asarray(Wh) + np.asarray(b)
    h_t = np.tanh(pre_act)
    return h_t
