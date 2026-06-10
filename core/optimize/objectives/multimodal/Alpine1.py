import numpy as np

class Alpine1:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))