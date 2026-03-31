def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # mu can be sliding window but sigma requires another sweep anyway
    n = len(values)
    k = window_size
    ans = []
    for i in range(n-k+1):
        mu = sum(values[i:i+k]) / k
        sigma = (sum((x-mu)**2 for x in values[i:i+k])/k) ** 0.5
        ans.append(sigma)

    return ans 

        