import numpy as np

class Salomon:
    BOUNDS = (-100, 100)

    def __call__(self, x):
        x = np.asarray(x)
        norm = np.sqrt(np.sum(x ** 2))
        
        return 1 - np.cos(2 * np.pi * norm) + 0.1 * norm