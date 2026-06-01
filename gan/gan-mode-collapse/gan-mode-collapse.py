import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    diversity_score = np.mean(np.std(np.asarray(generated_samples), axis=0))
    is_collapsed = diversity_score < threshold

    return {"diversity_score": diversity_score, "is_collapsed": is_collapsed}