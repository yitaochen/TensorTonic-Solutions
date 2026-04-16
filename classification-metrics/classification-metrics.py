import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    K = int(max(y_true.max(), y_pred.max())) + 1
    
    cm = np.bincount(y_true * K + y_pred, minlength=K**2).reshape(K, K)
    
    diag        = np.diag(cm)           # TP per class
    pred_support = cm.sum(axis=0)       # TP + FP per class
    true_support = cm.sum(axis=1)       # TP + FN per class
    total        = cm.sum()
    accuracy     = diag.sum() / total

    if average == "micro":
        TP = diag.sum()
        FP = (pred_support - diag).sum()
        FN = (true_support - diag).sum()
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = 2 / (1/precision + 1/recall)

    elif average == "macro":
        precision_cls = diag / pred_support
        recall_cls    = diag / true_support
        f1_cls        = 2 / (1/precision_cls + 1/recall_cls)
        precision, recall, f1 = precision_cls.mean(), recall_cls.mean(), f1_cls.mean()

    elif average == "weighted":
        weights       = true_support / total
        precision_cls = diag / pred_support
        recall_cls    = diag / true_support
        f1_cls        = 2 / (1/precision_cls + 1/recall_cls)
        precision = np.dot(weights, precision_cls)
        recall    = np.dot(weights, recall_cls)
        f1        = np.dot(weights, f1_cls)

    elif average == "binary":
        i         = pos_label
        precision = diag[i] / pred_support[i]
        recall    = diag[i] / true_support[i]
        f1        = 2 / (1/precision + 1/recall)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}