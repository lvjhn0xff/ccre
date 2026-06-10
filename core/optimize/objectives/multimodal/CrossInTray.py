import numpy as np

class CrossInTray:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        fact1 = np.sin(x[0]) * np.sin(x[1])
        fact2 = np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        
        return -0.0001 * (np.abs(fact1 * fact2) + 1) ** 0.1