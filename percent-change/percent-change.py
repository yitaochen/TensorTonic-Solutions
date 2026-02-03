def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    n = len(series)
    ans = [0.0] * (n - 1)
    for i in range(1, n):
        ans[i-1] = (series[i] - series[i-1]) / series[i-1] if series[i-1] != 0 else 0.0
    
    return ans 