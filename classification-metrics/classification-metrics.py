import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    # Build confusion matrix first 
    # for class i, TP = cm[i, i], FP = sum(cm[~i, i]), FN = sum(cm[i, ~i])
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    K = int(max(y_true.max(), y_pred.max())) + 1
    index = y_true * K + y_pred
    cm = np.bincount(index, minlength=K**2)
    cm = cm.reshape((K, K))
    pred_support = np.sum(cm, axis=0)
    true_support = np.sum(cm, axis=1)
    total = np.sum(cm)
    if average == "micro":
        TP, FP, FN = 0, 0, 0
        for i in range(K):
            TP += cm[i, i]
            FP += (pred_support[i] - cm[i, i])
            FN += (true_support[i] - cm[i, i])
        accuracy = TP / total 
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 / (1/precision + 1/recall)
    elif average == "macro":
        TP = np.zeros(K)
        FP = np.zeros(K)
        FN = np.zeros(K)
        for i in range(K):
            TP[i] = cm[i, i]
            FP[i] = pred_support[i] - cm[i, i]
            FN[i] = true_support[i] - cm[i, i]
        accuracy = np.sum(np.diag(cm)) / total
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 / (1/precision + 1/recall)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
    elif average == "weighted":
        TP = np.zeros(K)
        FP = np.zeros(K)
        FN = np.zeros(K)
        for i in range(K):
            TP[i] = cm[i, i]
            FP[i] = pred_support[i] - cm[i, i]
            FN[i] = true_support[i] - cm[i, i]
        weights = true_support / total  # true support fraction per class
        accuracy = np.sum(np.diag(cm)) / total
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = 2 / (1/precision + 1/recall)
        precision = np.sum(weights * precision)
        recall    = np.sum(weights * recall)
        f1        = np.sum(weights * f1)
    
    elif average == "binary":
        i = pos_label
        TP = cm[i, i]
        FP = pred_support[i] - cm[i, i]
        FN = true_support[i] - cm[i, i]
        accuracy  = np.sum(np.diag(cm)) / total
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = 2 / (1/precision + 1/recall)
        
        
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}