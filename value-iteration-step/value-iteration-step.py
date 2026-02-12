def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    num_states = len(values)
    num_actions = len(rewards[0])
    values_new = [0 for _ in range(num_states)]
    for s in range(num_states):
        values_new[s] = max(sum([rewards[s][a]] + [gamma*transitions[s][a][ns]*values[ns] for ns in range(num_states)]) for a in range(num_actions))
    
    return values_new