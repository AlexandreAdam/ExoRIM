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



def test_grad_likelihood():
    pixels = 64
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModelv1(pixels, mask_coordinates)

    x = np.arange(pixels) - pixels//2 + 0.5
    xx, yy = np.meshgrid(x, x)
    rho = np.hypot(xx, yy)
    image = np.zeros(shape=[1, pixels, pixels, 1])
    image[0, ..., 0] += np.exp(0.5 * rho**2/25)
    image /= image.sum()
    image = tf.constant(image, dtype=tf.float32)

    X = phys.forward(image)

    grad = phys.grad_log_likelihood(image, X).numpy()
    print(grad.max())
    print(grad.min())
    print(grad.mean())
    return grad


def test_grad_likelihood2():
    pixels = 128
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModelv1(pixels, mask_coordinates, lam=0)

    x = np.arange(pixels) - pixels//2 + 0.5
    xx, yy = np.meshgrid(x, x)
    rho1 = np.hypot(xx - 5, yy - 5)
    rho2 = np.hypot(xx + 5, yy + 5)
    image = np.zeros(shape=[1, pixels, pixels, 1])
    image[0, ..., 0] += np.exp(-0.5 * rho1**4/30)
    image[0, ..., 0] += np.exp(-0.5 * rho2**4/30)
    image /= image.sum()
    image = tf.constant(image, dtype=tf.float32)

    image2 = np.zeros_like(image)
    rho = np.hypot(xx, yy)
    image2[0, ..., 0] += np.exp(-0.5 * rho**4/30)
    image2 /= image2.sum()
    image2 = tf.constant(image2, tf.float32)

    X = phys.forward(image)

    grad = phys.grad_log_likelihood(image2, X).numpy()
    print(grad.max())
    print(grad.min())
    print(grad.mean())
    return image, image2, grad

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image, image2, grad = test_grad_likelihood2()
    plt.figure()
    plt.imshow(grad[0, ..., 0], origin="lower", cmap="hot")
    plt.title("Gradient")
    plt.colorbar()

    plt.figure()
    plt.imshow(image[0, ..., 0], origin="lower", cmap="hot")
    plt.title("Ground Truth")
    plt.colorbar()

    plt.figure()
    plt.imshow(image2[0, ..., 0], origin="lower", cmap="hot")
    plt.title("Guess")
    plt.colorbar()
    plt.show()




