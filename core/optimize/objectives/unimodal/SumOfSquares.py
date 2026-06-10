import numpy as np

class SumOfSquares:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        i = np.arange(1, len(x) + 1)
        
        return np.sum(i * x ** 2)