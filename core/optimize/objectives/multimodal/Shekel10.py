import numpy as np

class Shekel10:
    BOUNDS = (0, 10)

    def __call__(self, x):
        x = np.asarray(x)

        # Standard Shekel parameters (m = 10)
        a = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 5.0, 5.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6]
        ])

        c = np.array([
            0.1, 0.2, 0.2, 0.4, 0.4,
            0.6, 0.3, 0.7, 0.5, 0.5
        ])

        res = 0.0
        for i in range(10):
            dist_sq = np.sum((x - a[i, :len(x)]) ** 2)
            res += 1.0 / (dist_sq + c[i])

        return -res