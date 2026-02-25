def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    topk = set(recommended[:k])
    relevant = set(relevant)
    precision = len(topk & relevant) / k
    recall = len(topk & relevant) / len(relevant)

    return [precision, recall]