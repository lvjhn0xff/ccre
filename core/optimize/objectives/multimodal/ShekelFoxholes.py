import numpy as np

class ShekelFoxholes:
    BOUNDS = (-65.536, 65.536)

    def __call__(self, x):
        x = np.asarray(x)
        
        # De Jong's fifth function (Fifth Function of De Jong / Foxholes)
        # Typically defined in 2D with 25 foxholes
        a1 = np.tile([-32, -16, 0, 16, 32], 5)
        a2 = np.repeat([-32, -16, 0, 16, 32], 5)
        a = np.vstack((a1, a2))
        
        j = np.arange(1, 26)
        term = j + (x[0] - a[0, :]) ** 6 + (x[1] - a[1, :]) ** 6
        
        return 1.0 / (0.002 + np.sum(1.0 / term))