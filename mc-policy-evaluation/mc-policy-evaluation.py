import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    episodes = np.asarray(episodes)
    V = np.zeros((n_states, ))
    cnts = np.zeros((n_states, ))
    for eps in episodes:
        seen = np.zeros((n_states, ))
        G = 0
        for s, r in eps[::-1]:
            G = gamma * G + r
            if not seen[s]:
                V[s] += G 
                cnts[s] += 1
                seen[s] = 1
    
    return np.divide(V, cnts, out=np.zeros_like(V), where=cnts!=0)

