import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    def gini(count_k):
        p_k = count_k / sum(count_k)
        return 1 - np.sum(p_k**2)

    X = np.asarray(X)
    y = np.asarray(y)
    N, D = X.shape
    MAX = float("-inf")
    ans = []
    for j in range(D-1, -1, -1):
        unique_values= np.unique(X[:, j])
        _, counts = np.unique(y, return_counts=True)
        Gini_parent = gini(counts)
        m = len(unique_values)
        for i in range(m-1, 0, -1):
            midpoint = (unique_values[i] + unique_values[i-1]) / 2
            y_left = y[X[:, j]<=midpoint]
            y_right = y[X[:, j]>midpoint]
            N_left, N_right = len(y_left), len(y_right)
            if N_left == 0 or N_right == 0:
                continue 
            _, counts_left = np.unique(y_left, return_counts=True)
            _, counts_right = np.unique(y_right, return_counts=True)
            Gini_split = N_left / N * gini(counts_left) + N_right / N * gini(counts_right)
            if Gini_parent - Gini_split >= MAX:
                MAX = Gini_parent - Gini_split
                ans = [j, midpoint]

    return ans 
                
            
        
        