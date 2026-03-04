import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    if rng is not None:
        p = rng.random()
    else:
        p = np.random.rand()

    if epsilon == 0 or p >= epsilon:
        action = np.argmax(q_values)
    else:
        if rng is not None:
            action = rng.integers(low=0, high=len(q_values))
        else:
            action = np.random.randint(0, len(q_values))

    return action 
