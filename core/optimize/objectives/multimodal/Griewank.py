import numpy as np

class Griewank:
    BOUNDS = (-600, 600)

    def __call__(self, x):
        x = np.asarray(x)
        i = np.arange(1, len(x) + 1)
        
        term1 = np.sum(x ** 2) / 4000
        term2 = np.prod(np.cos(x / np.sqrt(i)))
        
        return term1 - term2 + 1