import numpy as np

class Easom:
    BOUNDS = (-100, 100)

    def __call__(self, x):
        x = np.asarray(x)
        
        return (
            -np.cos(x[0]) * 
            np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
        )