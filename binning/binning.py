def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    min_val, max_val = min(values), max(values)
    n = len(values)
    if min_val == max_val:
        return [0] * n 
    w = (max_val - min_val) / num_bins
    ans = [0] * n
    for i in range(n):
        bin_idx = int((values[i] - min_val) / w) 
        ans[i] = min(bin_idx, num_bins - 1)
        
    return ans 
        
    