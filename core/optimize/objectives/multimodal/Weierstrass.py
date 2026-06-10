import numpy as np

class Weierstrass:
    BOUNDS = (-0.5, 0.5)

    def __call__(self, x, a=0.5, b=3, k_max=20):
        x = np.asarray(x)
        n = len(x)
        
        k = np.arange(0, k_max + 1)
        a_k = a ** k
        b_k = b ** k
        
        # Calculate the main sum over all dimensions
        term1 = 0.0
        for xi in x:
            term1 += np.sum(a_k * np.cos(2 * np.pi * b_k * (xi + 0.5)))
            
        # Calculate the constant regularization term
        term2 = n * np.sum(a_k * np.cos(np.pi * b_k))
        
        return term1 - term2