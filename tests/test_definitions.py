from __future__ import division

from ExoRIM.definitions import *
from ExoRIM.operators import NDFTM, Baselines
import numpy as np
import tensorflow as tf
import time


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


def test_chisq_vis_gradient():
    pixels = 32
    mask = np.random.normal(0, 6, [21, 2])
    B = Baselines(mask)
    A = tf.constant(NDFTM(B.UVC, 0.5e-6, pixels, 0.3), mycomplex)
    sigma = 0.01

    image = np.zeros([pixels, pixels])
    blob_size = 4
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + yy**2)
    image += np.exp(-rho**2/blob_size**2)

    rho_prime = np.sqrt((xx - 10)**2 + (yy - 10)**2)
    noise = np.zeros_like(image)
    noise += np.exp(-rho_prime**2/blob_size**2)
    noise = tf.constant(noise.reshape((1, pixels, pixels)), dtype)

    vis = tf.constant(np.dot(A, image.flatten()).reshape((1, -1)), mycomplex)

    start = time.time()
    grad1 = chisqgrad_vis_auto(noise, A, vis, sigma)
    print(f"Auto grad took {time.time() - start:.3f} sec")
    start = time.time()
    grad2 = chisqgrad_vis(noise, A, vis, sigma)
    print(f"Analytical grad took {time.time() - start:.3f} sec")
    assert np.allclose(sigma**2*grad1.numpy(), sigma**2*grad2.numpy(), rtol=1e-3)

if __name__ == '__main__':
    test_chisq_vis_gradient()