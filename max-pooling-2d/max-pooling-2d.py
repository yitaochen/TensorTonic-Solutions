def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    H, W = len(X), len(X[0])
    H_out = H // pool_size
    W_out = W // pool_size
    ans = [[0 for _ in range(W_out)] for _ in range(H_out)]
    for i in range(H_out):
      for j in range(W_out):
        ans[i][j] = max(X[i*pool_size+a][j*pool_size+b] for a in range(pool_size) for b in range(pool_size))

    return ans 
        