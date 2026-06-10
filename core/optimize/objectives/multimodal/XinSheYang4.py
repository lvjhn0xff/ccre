import numpy as np

class XinSheYang4:
    BOUNDS = (-10, 10)

    def __call__(self, x):
        x = np.asarray(x)
        
        term1 = np.sum(np.sin(x) ** 2)
        term2 = np.exp(-np.sum(x ** 2))
        term3 = np.exp(-np.sum(np.sin(np.sqrt(np.abs(x))) ** 2))
        
        return (term1 - term2) * term3