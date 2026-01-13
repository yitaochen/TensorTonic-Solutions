import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    # Ensure input is a numpy array (and float to avoid integer division issues)
    C = np.asarray(C, dtype=float)
    
    # 1. Calculate totals
    grand_total = np.sum(C)
    row_totals = np.sum(C, axis=1)
    col_totals = np.sum(C, axis=0)
    
    # 2. Compute Expected Frequencies Matrix
    # We use the outer product of row and column totals divided by the grand total
    # This creates the matrix where E_ij = (row_i * col_j) / total
    expected = np.outer(row_totals, col_totals) / grand_total
    
    # 3. Compute Chi-Square Statistic
    # Vectorized calculation: sum((O - E)^2 / E)
    chi2 = np.sum((C - expected) ** 2 / expected)
    
    return chi2, expected