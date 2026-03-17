def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    T = len(rewards)
    G = [0] * T
    G[-1] = rewards[-1]
    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]
    mu = sum(G) / T 
    A = [g - mu for g in G]
    L = -sum(pr*a for pr, a in zip(log_probs, A)) / T 

    return L 