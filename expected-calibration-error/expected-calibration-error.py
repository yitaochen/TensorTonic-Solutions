def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    acc = [0.0] * n_bins
    conf = [0.0] * n_bins
    cnt_pred = [0] * n_bins
    n = len(y_true)
    for t, p in zip(y_true, y_pred):
        if p == 1.0:
            bin_idx = -1
        else:
            bin_idx = int(p * n_bins)
        conf[bin_idx] += p
        cnt_pred[bin_idx] += 1
        acc[bin_idx] += t
    conf = [p/cnt if cnt > 0 else 0.0 for p, cnt in zip(conf, cnt_pred)]
    acc = [t/cnt if cnt > 0 else 0.0 for t, cnt in zip(acc, cnt_pred)]

    ans = 0
    for t_avg, p_avg, b_cnt in zip(acc, conf, cnt_pred):
        ans += (b_cnt / n) * abs(t_avg - p_avg)
        
    return ans 
        