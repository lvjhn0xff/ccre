import numpy as np

class Ackley:
    BOUNDS = (-32.768, 32.768)

    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        
        return term1 + term2 + 20 + np.e