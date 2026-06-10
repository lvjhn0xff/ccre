import numpy as np

class Rastrigin:
    BOUNDS = (-5.12, 5.12)

    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))