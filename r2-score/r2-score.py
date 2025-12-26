import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    # handle edge case 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_mean = np.mean(y_true, keepdims=True)
    if (y_true == y_true[0]).all():
      if (y_true == y_pred).all():
        return 1.0
      else:
        return 0.0

    r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_mean - y_true) ** 2)

    return r2