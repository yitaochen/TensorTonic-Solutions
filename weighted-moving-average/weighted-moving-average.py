def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    w_sum = sum(weights)
    k = len(weights)
    ans = [0] * (len(values) - len(weights) + 1)
    for i in range(len(ans)):
        ans[i] = sum(weights[j]*values[i+j] for j in range(k)) / w_sum

    return ans 