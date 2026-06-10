import numpy as np

class Rosenbrock:
    BOUNDS = (-2.048, 2.048)

    def __call__(self, x):
        x = np.asarray(x)
        
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)