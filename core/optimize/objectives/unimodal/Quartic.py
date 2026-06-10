import numpy as np

class Quartic:
    BOUNDS = (-1.28, 1.28)

    def __call__(self, x):
        x = np.asarray(x)
        i = np.arange(1, len(x) + 1)
        
        return np.sum(i * x ** 4) + np.random.randn()