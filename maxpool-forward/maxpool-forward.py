def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H, W = len(X), len(X[0])
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    ans = [[0 for _ in range(W_out)] for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            ans[i][j] = max(X[m][n] for m in range(stride * i, stride * i + pool_size) for n in range(stride * j, stride * j + pool_size))
    
    return ans 
