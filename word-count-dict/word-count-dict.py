def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    ans = {}
    for s in sentences:
        for w in s:
            ans[w] = ans.get(w, 0) + 1

    return ans 
    