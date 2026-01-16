import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    if len(fpr) != len(tpr) or len(fpr) < 2 or len(tpr) < 2:
        return 0.0
    return float(np.trapezoid(y=tpr, x=fpr))