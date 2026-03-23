import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x=np.array(x)
    p=np.array(p)
    sum=0
    if not np.isclose(np.sum(p),1):
        raise ValueError("Probabilities Must sum to 1")
    for i in range(len(x)):
        sum+=x[i]*p[i]
        
    return sum
        
    pass
