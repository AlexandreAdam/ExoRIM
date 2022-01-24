from exorim.physical_model import PhysicalModel, GOLAY9, JWST_NIRISS_MASK
from exorim.simulated_data import CenteredBinariesDataset
from exorim.operators import Baselines
from exorim.definitions import rad2mas
import tensorflow as tf
import numpy as np


def test_nyquist_sampling_criterion():
    pixels = 32
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates)
    # get frequency sampled in Fourier space
    B = Baselines(mask_coordinates)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    freq_sampled = 1/rad2mas(1/rho)
    sampling_frequency = 1/phys.plate_scale # 1/mas


    print(freq_sampled.max())
    print(sampling_frequency)
    assert sampling_frequency > 2 * freq_sampled.max()

    phys = PhysicalModel(pixels, GOLAY9) # GOLAY9 mask
    # get frequency sampled in Fourier space
    B = Baselines(GOLAY9)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    freq_sampled = 1/rad2mas(1/rho)
    sampling_frequency = 1/phys.plate_scale # 1/mas


    print(freq_sampled.max())
    print(sampling_frequency)
    assert np.round(sampling_frequency, 5) > np.round(2 * freq_sampled.max(), 5)


def test_fov():
    pixels = 32
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates)
    # get frequency sampled in Fourier space
    B = Baselines(mask_coordinates)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    fov = rad2mas(1/rho.min())
    reconstruction_fov = pixels * phys.plate_scale
    assert np.round(fov, 5) <= np.round(reconstruction_fov, 5) # reconstruction should at least encapsulate largest scale



def test_grad_likelihood():
    pixels = 32
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates)

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
    pixels = 32
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates)

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
    # image, image2, grad = test_grad_likelihood2()
    # plt.figure()
    # plt.imshow(grad[0, ..., 0], origin="lower", cmap="hot")
    # plt.title("Gradient")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(image[0, ..., 0], origin="lower", cmap="hot")
    # plt.title("Ground Truth")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(image2[0, ..., 0], origin="lower", cmap="hot")
    # plt.title("Guess")
    # plt.colorbar()
    # plt.show()

    phys = PhysicalModel(pixels=32, mask_coordinates=JWST_NIRISS_MASK, analytic=True, wavelength=3.8e-6)
    dataset = CenteredBinariesDataset(phys, total_items=1, batch_size=1, width=2)
    X, image = dataset.generate_batch()
    plt.imshow(image[0, ..., 0])
    plt.show()

    plt.figure()
    fft = np.abs(np.fft.fftshift(np.fft.fft2(image[0, ..., 0])))
    pixels = 32
    wavel = 8e-6
    uv = phys.baselines.UVC
    rho = np.hypot(uv[:, 0], uv[:, 1])
    fov = rad2mas(wavel / rho).max()
    plate_scale = fov / pixels  # mas
    fftfreq = np.fft.fftshift(np.fft.fftfreq(pixels, plate_scale))

    im = plt.imshow(np.abs(fft), cmap="hot", extent=[fftfreq.min(), fftfreq.max()] * 2)
    baselines = Baselines(mask_coordinates=GOLAY9)
    ufreq = 1 / rad2mas(1 / baselines.UVC[:, 0] * wavel)
    vfreq = 1 / rad2mas(1 / baselines.UVC[:, 1] * wavel)
    plt.plot(ufreq, vfreq, "bo")
    plt.colorbar(im)
    plt.title("UV coverage")
    plt.show()
