import numpy as np
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x=x0
    def function(x):
        return a*x**2+b*x+c
    def f_prime(x):
        return 2*a*x+b
    
    for i in range(steps):
        grad=f_prime(x)
        x=x-lr*grad #Gradient descent update equation
        
    return x
        
    pass