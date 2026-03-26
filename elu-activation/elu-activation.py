def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    return [alpha * (math.exp(a) - 1) if a <= 0 else a for a in x]
            
    