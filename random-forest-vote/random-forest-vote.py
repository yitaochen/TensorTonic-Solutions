import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    ans = []
    for pred in zip(*predictions):
        vals, cnts = np.unique(pred, return_counts=True)
        max_f = max(cnts)
        ans.append(vals[cnts==max_f][0])

    return ans 