import numpy as np

class Schwefel226:
    BOUNDS = (-500, 500)

    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))