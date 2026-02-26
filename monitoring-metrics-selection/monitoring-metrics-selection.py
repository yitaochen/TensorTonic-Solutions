def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    n = len(y_true)
    ans = []
    if system_type == "classification":
        TP = sum(1 if y_pred[i] == 1 and y_true[i] == 1 else 0 for i in range(n))
        TN = sum(1 if y_pred[i] == 0 and y_true[i] == 0 else 0 for i in range(n))
        FP = sum(1 if y_pred[i] == 1 and y_true[i] == 0 else 0 for i in range(n))
        FN = sum(1 if y_pred[i] == 0 and y_true[i] == 1 else 0 for i in range(n))
        accuracy = (TP + TN) / n
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        ans.append(("accuracy", accuracy))
        ans.append(("precision", precision))
        ans.append(("recall", recall))
        ans.append(("f1", f1))
    elif system_type == "regression":
        mae = sum(abs(y_pred[i]-y_true[i]) for i in range(n))/n
        rmse = (sum((y_pred[i]-y_true[i])**2 for i in range(n))/n)**0.5
        ans.append(("mae", mae))
        ans.append(("rmse", rmse))
    elif system_type == "ranking":
        recommended_and_relevant = list(zip(y_pred, y_true))
        recommended_and_relevant.sort(reverse=True)
        relevant_in_top_3 = sum(recommended_and_relevant[i][1] for i in range(3))
        precision_at_3 = relevant_in_top_3 / 3
        recall_at_3 = relevant_in_top_3 / sum(y_true) if sum(y_true) > 0 else 0.0
        ans.append(("precision_at_3", precision_at_3))
        ans.append(("recall_at_3", recall_at_3))
        

    ans.sort(key=lambda x: x[0])
    return ans 
        
        