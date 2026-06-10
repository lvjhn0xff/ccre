import numpy as np

class EggHolder:
    BOUNDS = (-512, 512)

    def __call__(self, x):
        x = np.asarray(x)
        
        term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47))))
        term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        
        return term1 + term2