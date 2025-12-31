import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    x = np.asarray(matrix, dtype=float)
    
    if x.ndim != 2:
        return None

    if axis is not None and (axis >= x.ndim or axis < -x.ndim):
        return None

    if norm_type == "l2":
        norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    elif norm_type == "l1":
        norm = np.sum(np.abs(x), axis=axis, keepdims=True)
    elif norm_type == "max":
        norm = np.max(np.abs(x), axis=axis, keepdims=True)
    else:
        return None


    result = np.divide(x, norm, out=np.zeros_like(x), where=norm!=0)
    
    return result