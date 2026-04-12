import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    index = y_true * num_classes + y_pred
    cm = np.bincount(index, minlength=num_classes**2)
    cm = cm.reshape((num_classes, num_classes))

    if normalize == 'none':
        return cm 

    cm = cm.astype(np.float64)
    if normalize == 'true':
        row_sums = np.sum(cm, axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)
    elif normalize == 'pred':
        col_sums = np.sum(cm, axis=0, keepdims=True)
        cm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums!=0)
    elif normalize == 'all':
        total = np.sum(cm)
        cm = cm / total if total > 0 else cm 

    return cm 
    
        

    
        
        