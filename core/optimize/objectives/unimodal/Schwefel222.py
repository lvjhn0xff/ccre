import numpy as np

class Schwefel222:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        
        return np.sum(abs_x) + np.prod(abs_x)