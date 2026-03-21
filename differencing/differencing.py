def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    input = series[:]
    for i in range(1, order+1):
        n = len(input)
        ans = [0] * (n - 1)
        for i in range(n-1):
            ans[i] = input[i+1] - input[i]
        input = ans[:]

    return ans 
        
        
        