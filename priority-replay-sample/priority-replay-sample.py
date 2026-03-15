def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    n = len(priorities)
    powered = [p**alpha for p in priorities]
    total = sum(powered)
    probs = [x/total for x in powered]
    raw_weights = [(n*pr)**(-beta) for pr in probs]
    max_weight = max(raw_weights)
    normalized_weights = [w/max_weight for w in raw_weights]

    return [probs, normalized_weights]