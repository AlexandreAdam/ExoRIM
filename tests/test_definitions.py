from __future__ import division

from ExoRIM.definitions import pixel_grid
import numpy as np


def test_pixel_grid():
    pixels = 32
    xx, yy = pixel_grid(pixels, symmetric=True)
    # assert symmetry along x dimension
    assert np.all(xx == -xx[:, ::-1])
    # assert symmetry along y dimension
    assert np.all(yy == -yy[::-1])
    pixels = 33
    xx, yy = pixel_grid(pixels, symmetric=True)
    assert np.all(xx == -xx[:, ::-1])
    assert np.all(yy == -yy[::-1])


