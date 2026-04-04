def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    L = (6 / fan_in) ** 0.5
    m, n = len(W), len(W[0])

    return [[W[i][j]*2*L - L for j in range(n)] for i in range(m)]