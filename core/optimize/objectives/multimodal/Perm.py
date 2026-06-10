import numpy as np

class Perm:
    BOUNDS = (-1, 1) # Commonly optimized for d = len(x), beta = 0.5 within [-d, d] or [-1, 1]

    def __call__(self, x, beta=0.5):
        x = np.asarray(x)
        d = len(x)
        j = np.arange(1, d + 1)
        
        outer_sum = 0.0
        for i in range(1, d + 1):
            inner_sum = np.sum((j ** i + beta) * ((x / j) ** i - 1))
            outer_sum += inner_sum ** 2
            
        return outer_sum