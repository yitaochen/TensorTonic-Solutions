import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # ── 1. Sort by descending score; break ties by putting positives first ──
    # np.lexsort sorts by the LAST key first, so:
    #   primary key   = y_score  (descending → negate)
    #   secondary key = y_true   (descending → negate, so positives come first on ties)
    sorted_indices = np.lexsort((-y_true, -y_score))
    y_true_sorted  = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]

    # ── 2. Cumulative true positives at every possible threshold position ──
    cum_tp = np.cumsum(y_true_sorted)           # TP count after each sample
    cum_fp = np.arange(1, len(y_true) + 1) - cum_tp  # FP = total seen − TP

    total_pos = cum_tp[-1]                      # all positives in dataset
    total_neg = len(y_true) - total_pos         # all negatives in dataset

    # ── 3. Keep only positions where the threshold actually changes ──
    # A new threshold occurs wherever the score changes to a lower value.
    # np.diff finds those transitions; np.where returns their indices.
    threshold_indices = np.where(np.diff(y_score_sorted, append=-np.inf))[0]

    tpr        = cum_tp[threshold_indices] / total_pos
    fpr        = cum_fp[threshold_indices] / total_neg
    thresholds = y_score_sorted[threshold_indices]

    # ── 4. Prepend the (0, 0) origin — no samples classified positive yet ──
    tpr        = np.concatenate(([0.0], tpr))
    fpr        = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([np.inf], thresholds))

    return fpr, tpr, thresholds