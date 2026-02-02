

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    m, n = len(X), len(W[0])
    Y = [[0 for _ in range(n)] for _ in range(m)]

    def dot_product(a, b):
        if len(a) != len(b):
            raise ValueError("Lists must have the same length")
        ans = 0
        for i in range(len(a)):
            ans += a[i] * b[i]
        return ans 
    
    W_cols = list(zip(*W))
    
    # print(W, W_cols)
    # print(m, n)
    for i in range(m):
        for j in range(n):
            Y[i][j] = dot_product(X[i], W_cols[j]) + b[j] 
    # print(X, W, b, Y)
    return Y