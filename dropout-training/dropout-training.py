import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.array(x,dtype=float)
    if rng is None:
        rng=np.random.default_rng()
        
    rand_vals = rng.random(x.shape)  
    #random bw 0 and 1 in the same shape as x

    pattern =(rand_vals<(1-p))*(1.0/(1-p))

    #apply
    out=x*pattern

    return out,pattern
    
    pass