import numpy as np

class DropWave:
    BOUNDS = (-5.12, 5.12)

    def __call__(self, x):
        x = np.asarray(x)
        sum_sq = np.sum(x ** 2)
        
        numerator = 1 + np.cos(12 * np.sqrt(sum_sq))
        denominator = 0.5 * sum_sq + 2
        
        return -numerator / denominator