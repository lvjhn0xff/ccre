import numpy as np

class Sphere:
    BOUNDS = (-5.12, 5.12)

    def __call__(self, x):
        x = np.asarray(x)
        
        return np.sum(x ** 2)