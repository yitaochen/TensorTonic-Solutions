def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    unrated_items = [(s, i) for i, s in enumerate(scores) if i not in rated_indices]
    ans = [i for s, i in sorted(unrated_items, key=lambda x: (x[0], -x[1]), reverse=True)]
    return ans[:k]