import numpy as np

class Elliptic:
    BOUNDS = (-100, 100)

    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        a = 10 ** 6
        i = np.arange(n)
        
        return np.sum(a ** (i / (n - 1)) * x ** 2)