import numpy as np

class Zakharov:
    BOUNDS = (-5, 10)

    def __call__(self, x):
        x = np.asarray(x)
        i = np.arange(1, len(x) + 1)
        term = 0.5 * np.sum(i * x)
        
        return np.sum(x ** 2) + term ** 2 + term ** 4