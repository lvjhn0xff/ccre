import numpy as np

class Schwefel221:
    BOUNDS = (-100, 100)

    def __call__(self, x):
        x = np.asarray(x)
        
        return np.max(np.abs(x))