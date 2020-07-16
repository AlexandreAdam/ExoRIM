from ExoRIM.operators import Baselines
import numpy as np


def test_baselines():
    N = 11
    L = 100
    var = 10
    x = (L + np.random.normal(0, var, N)) * np.cos(2 * np.pi * np.arange(N) / N)
    y = (L + np.random.normal(0, var, N)) * np.sin(2 * np.pi * np.arange(N) / N)
    circle_mask = np.array([x, y]).T
    B = Baselines(circle_mask)
