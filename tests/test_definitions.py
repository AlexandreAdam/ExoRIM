from __future__ import division

from ExoRIM.definitions import pixel_grid, softmax_scaler, dtype, INTENSITY_SCALER
import numpy as np
import tensorflow as tf


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


def test_softmax_scaler():
    x = tf.constant(np.linspace(-100, 100, 100), dtype)
    alpha = INTENSITY_SCALER
    estimated_max = tf.einsum("i, i ->", x, tf.math.softmax(alpha * x))
    assert abs(estimated_max.numpy() - 100) < 0.0001, f"Softmax avec alpha = {alpha} est une mauvaise estimation du maximum"
    renormalized_x = softmax_scaler(x, -1, 1).numpy()
    print(renormalized_x)
    assert abs(renormalized_x.min() + 1) < 0.0001, f"Softmax scaler gives lower bound of range {renormalized_x.min()} instead of -1"
    assert abs(renormalized_x.max() - 1) < 0.0001, f"Softmax scaler gives upper bound of range {renormalized_x.min()} instead of 1"
    renormalized_x = softmax_scaler(x, 0, 100).numpy()
    print(renormalized_x)
    assert abs(renormalized_x.min()) < 0.0001, f"Softmax scaler gives lower bound of range {renormalized_x.min()} instead of 0"
    assert abs(renormalized_x.max() - 100) < 0.0001, f"Softmax scaler gives upper bound of range {renormalized_x.min()} instead of 100"
