import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("p should sum up to 1.0")
    # if len(x) != len(p):
    #     return ValueError("x and p should match on length!")

    return np.sum(x * p)