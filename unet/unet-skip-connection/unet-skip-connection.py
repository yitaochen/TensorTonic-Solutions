import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    B, H_e, W_e, C_e = encoder_features.shape
    _, H_d, W_d, C_d = decoder_features.shape 
    diff_H = H_e - H_d
    start_H = diff_H // 2
    diff_W = W_e - W_d 
    start_W = diff_W // 2

    return np.concatenate((encoder_features[:, start_H:start_H+H_d, start_W:start_W+W_d, :], decoder_features), axis=3)
