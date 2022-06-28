from exorim.physical_model import GOLAY9, JWST_NIRISS_MASK
from exorim.datasets.centered_binary_toy_dataset import CenteredBinariesDataset
from exorim.definitions import general_gamma_binary, super_gaussian
from exorim import PhysicalModel, Operators
from exorim.definitions import rad2mas
import tensorflow as tf
import numpy as np


def test_binary():
    pixels = 248
    sep = 86  # mas
    PA = 102  # degrees
    contrast = 5
    wavel = 1.7e-6  # hband, see Keck, Palomar

    phys = PhysicalModel(pixels, GOLAY9, wavel)
    B = phys.operators
    V1, V2, V3 = phys.V1.numpy(), phys.V2.numpy(), phys.V3.numpy()

    plate_scale = phys.plate_scale
    image = np.zeros((pixels, pixels))
    image += super_gaussian(pixels, plate_scale, sep/2, PA, 5) / contrast
    image += super_gaussian(pixels, plate_scale, -sep/2, PA, 5)

    # Compute the bispectrum and closure phase using our Physical Model
    phase = np.angle(phys.visibility(image[None, ..., None]))[0]
    bis = phys.bispectrum(image[None, ..., None])[0]
    cp1 = np.angle(bis)

    # Mathematical model for 2 delta functions
    gamma_model = general_gamma_binary(B.UVC, wavel, sep, PA, contrast)
    phase_model = np.angle(gamma_model)
    g1 = V1 @ gamma_model
    g2 = V2 @ gamma_model
    g3 = V3 @ gamma_model
    bis_model = g1 * g2.conj() * g3
    cp_model = np.angle(bis_model)

    # print(phase)
    # print(phase_model)
    assert np.allclose(phase, phase_model, rtol=1e-5)
    # print(cp1)
    # print(cp_model)
    assert np.allclose(cp1, cp_model, rtol=1e-4)


def test_nyquist_sampling_criterion():
    pixels = 32
    wavel = 3.8e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates, wavel, oversampling_factor=2)
    # get frequency sampled in Fourier space
    B = Operators(mask_coordinates, wavel)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    freq_sampled = 1/rad2mas(1/rho)
    sampling_frequency = 1/phys.plate_scale # 1/mas

    print(freq_sampled.max())
    print(sampling_frequency)
    assert sampling_frequency > 2 * freq_sampled.max()

    phys = PhysicalModel(pixels, GOLAY9) # GOLAY9 mask
    # get frequency sampled in Fourier space
    B = Operators(GOLAY9, wavel)
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
    phys = PhysicalModel(pixels, mask_coordinates, oversampling_factor=2)
    # get frequency sampled in Fourier space
    B = Operators(mask_coordinates, wavel)
    uv = B.UVC/wavel
    rho = np.hypot(uv[:, 0], uv[:, 1])
    fov = rad2mas(1/rho.min())
    reconstruction_fov = pixels * phys.plate_scale
    assert np.round(fov, 5) <= np.round(reconstruction_fov, 5) # reconstruction should at least encapsulate largest scale


def test_grad_likelihood():
    pixels = 32
    wavel = 0.5e-6
    mask_coordinates = tf.random.normal((12, 2))
    phys = PhysicalModel(pixels, mask_coordinates, wavel, oversampling_factor=2)

    x = np.arange(pixels) - pixels//2 + 0.5
    xx, yy = np.meshgrid(x, x)
    rho = np.hypot(xx, yy)
    image = np.zeros(shape=[1, pixels, pixels, 1])
    image[0, ..., 0] += np.exp(0.5 * rho**2/25)
    image /= image.sum()
    image = tf.constant(image, dtype=tf.float32)

    X = phys.forward(image)
    sigma = tf.ones_like(X) * 1e-2

    grad, chi_squared = phys.grad_chi_squared(image, X, sigma)
    grad = grad.numpy()
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
    sigma = tf.ones_like(X) * 1e-2
    grad, chi_squared = phys.grad_chi_squared(image2, X, sigma)
    grad = grad.numpy()
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

    pixels = 32
    wavel = 3.8e-6
    phys = PhysicalModel(pixels=pixels, mask_coordinates=JWST_NIRISS_MASK, wavelength=wavel, oversampling_factor=2)
    dataset = CenteredBinariesDataset(phys, total_items=1, batch_size=1, width=2)
    X, image = dataset.generate_batch(1)
    plt.imshow(image[0, ..., 0])
    plt.show()

    plt.figure()
    fft = np.abs(np.fft.fftshift(np.fft.fft2(image[0, ..., 0])))
    uv = phys.operators.UVC
    rho = np.hypot(uv[:, 0], uv[:, 1])
    fftfreq = np.fft.fftshift(np.fft.fftfreq(pixels, phys.plate_scale))

    im = plt.imshow(np.abs(fft), cmap="hot", extent=[fftfreq.min(), fftfreq.max()] * 2)
    ufreq = 1 / rad2mas(1 / uv[:, 0] * wavel)
    vfreq = 1 / rad2mas(1 / uv[:, 1] * wavel)
    plt.plot(ufreq, vfreq, "bo")
    plt.colorbar(im)
    plt.title("UV coverage")
    plt.show()
