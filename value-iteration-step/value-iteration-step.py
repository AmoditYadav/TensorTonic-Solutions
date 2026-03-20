def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states=len(values)
    num_actions=len(transitions[0])
    new_values=[0.0]*num_states
    
    for s in range(num_states): #states
        max_q=float('-inf')
        for a in range(num_actions): #loop over actions
            q=rewards[s][a]+gamma*sum(
                transitions[s][a][s_next]*values[s_next] for s_next in range(num_states)
                
            )
            max_q=max(max_q,q)
        new_values[s]=max_q
    return new_values
        
    