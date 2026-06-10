import numpy as np

class DixonPrice:
    BOUNDS = (-10, 10)
    
    def __call__(self, x):
        x = np.asarray(x)
        term1 = (x[0] - 1) ** 2
        
        i = np.arange(2, len(x) + 1)
        term2 = np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2)
        
        return term1 + term2