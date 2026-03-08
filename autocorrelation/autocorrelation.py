def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    n = len(series)
    mu = sum(series) / n
    gamma0 = sum((x-mu)**2 for x in series)
    if gamma0 == 0:
        return [1.0] + [0.0] * max_lag
    ans = [1.0] * (1 + max_lag)
    for lag in range(1, max_lag+1):
        ans[lag] = sum((series[t]-mu)*(series[t+lag]-mu) for t in range(n-lag)) / gamma0

    return ans 
        
    