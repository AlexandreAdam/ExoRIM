from ExoRIM.physical_model import PhysicalModelv1
from ExoRIM.operators import Baselines
from ExoRIM.definitions import rad2mas
import tensorflow as tf
import numpy as np


def test_nyquist_sampling_criterion():
    pixels = 64
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModelv1(pixels, mask_coordinates)
    # get frequency sampled in Fourier space
    B = Baselines(mask_coordinates)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    freq_sampled = 1/rad2mas(1/rho)
    sampling_frequency = 1/phys.plate_scale # 1/mas


    print(freq_sampled.max())
    print(sampling_frequency)
    assert sampling_frequency > 2 * freq_sampled.max()


def test_fov():
    pixels = 64
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModelv1(pixels, mask_coordinates)
    # get frequency sampled in Fourier space
    B = Baselines(mask_coordinates)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    fov = rad2mas(1/rho.min())
    reconstruction_fov = pixels * phys.plate_scale
    assert fov <= reconstruction_fov # reconstruction should at least encapsulate largest scale

