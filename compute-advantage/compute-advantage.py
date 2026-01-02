import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    T = len(states)
    Gt = 0
    A = [0] * T

    for i in range(T)[::-1]:
        Gt = rewards[i] + gamma * Gt 
        Vt = V[states[i]]
        A[i] = Gt - Vt 
    
    return np.asarray(A) 



