import numpy as np

class Michalewicz:
    BOUNDS = (0, np.pi)

    def __call__(self, x, m=10):
        x = np.asarray(x)
        i = np.arange(1, len(x) + 1)
        
        return -np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m))