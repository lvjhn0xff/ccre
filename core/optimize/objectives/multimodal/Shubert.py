import numpy as np

class Shubert:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        
        # Typically evaluated for 2 dimensions, but can generalize to n dimensions
        res = 1.0
        for xi in x:
            i = np.arange(1, 6)
            inner_sum = np.sum(i * np.cos((i + 1) * xi + i))
            res *= inner_sum
            
        return res