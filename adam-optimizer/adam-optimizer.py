import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #converting to numpy array for safe compute
    param = np.array(param, dtype=float) 
    grad = np.array(grad, dtype=float) 
    m = np.array(m, dtype=float) 
    v = np.array(v, dtype=float)
    
    m_t=beta1*m+(1-beta1)*grad
    v_t=beta2*v+(1-beta2)*(grad**2)
    #bias correction
    m_bias=m_t/(1-beta1**t)
    v_bias=v_t/(1-beta2**t)
    #param update
    param_new=param-lr*(m_bias/(np.sqrt(v_bias)+eps))

    return param_new, m_t, v_t
    pass