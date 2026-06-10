import numpy as np

class Step:
    BOUNDS = (-100, 100)

    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(np.floor(x + 0.5) ** 2)