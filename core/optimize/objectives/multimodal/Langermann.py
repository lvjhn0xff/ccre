import numpy as np

class Langermann:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        fact1 = np.sin(x[0]) * np.cos(x[1])
        fact2 = np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        
        return -np.abs(fact1 * fact2)