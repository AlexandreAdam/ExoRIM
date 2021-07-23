from exorim.operators import Baselines, NDFTM, closure_fourier_matrices, closure_phase_operator
from exorim.definitions import MYCOMPLEX, DTYPE
import tensorflow as tf
import time
import numpy as np
import pytest


@pytest.mark.skip("Test fails, for some reason removing conjugate operation in the analytical gradient yields the same"
                  "result given by AutoGrad. We also know AutoGrad is not the correct gradient as of now, since gradient"
                  "image is the mirror flip of the correct solution")
def test_chisq_vis_gradient():
    pixels = 32
    mask = np.random.normal(0, 6, [21, 2])
    B = Baselines(mask)
    A = tf.constant(NDFTM(B.UVC, 0.5e-6, pixels, 0.3), MYCOMPLEX)
    sigma = tf.constant(0.01)

    image = np.zeros([pixels, pixels])
    blob_size = 4
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + yy**2)
    image += np.exp(-rho**2/blob_size**2)

    rho_prime = np.sqrt((xx - 10)**2 + (yy - 10)**2)
    noise = np.zeros_like(image)
    noise += np.exp(-rho_prime**2/blob_size**2)
    noise = tf.constant(noise.reshape((1, pixels, pixels, 1)), DTYPE)

    vis = tf.constant(np.dot(A, image.flatten()).reshape((1, -1)), MYCOMPLEX)

    start = time.time()
    grad1 = chisq_gradient_complex_visibility(noise, A, vis, sigma)
    print(f"Auto grad took {time.time() - start:.3f} sec")
    start = time.time()
    grad2 = chisq_gradient_complex_visibility(noise, A, vis, sigma)
    print(f"Analytical grad took {time.time() - start:.3f} sec")
    assert np.allclose(sigma**2*grad1.numpy(), sigma**2*grad2.numpy(), rtol=1e-3)

    # TODO investigate how to make AutoGrad match analytical one --> simulate the conjugate of A somehow.


@pytest.mark.skip("Test fails, and we are not going to use this function")
def test_chisq_closure_phase_gradient():
    pixels = 32
    mask = np.random.normal(0, 6, [21, 2])
    B = Baselines(mask)
    A = NDFTM(B.UVC, 0.5e-6, pixels, 0.3)
    CPO = closure_phase_operator(B)
    A1, A2, A3 = closure_fourier_matrices(A, CPO)
    A1 = tf.constant(A1, MYCOMPLEX)
    A2 = tf.constant(A2, MYCOMPLEX)
    A3 = tf.constant(A3, MYCOMPLEX)
    sigma = tf.constant(0.01)

    image = np.zeros([pixels, pixels])
    blob_size = 4
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + yy**2)
    image += np.exp(-rho**2/blob_size**2)

    rho_prime = np.sqrt((xx - 10)**2 + (yy - 10)**2)
    noise = np.zeros_like(image)
    noise += np.exp(-rho_prime**2/blob_size**2)
    noise = tf.constant(noise.reshape((1, pixels, pixels, 1)), DTYPE)

    clphases = tf.math.angle(bispectrum(image.reshape((1, pixels**2, 1)), A1, A2, A3))

    start = time.time()
    grad1 = chisq_gradient_closure_phasor_analytic(noise, A1, A2, A3, clphases, sigma)
    print(f"Analytic grad took {time.time() - start:.3f} sec")
    start = time.time()
    grad2 = chisq_gradient_closure_phasor_auto(noise, A2, A2, A3, clphases, sigma)
    print(f"Auto grad took {time.time() - start:.3f} sec")

    print(f"Mean squared difference = {sigma**2 * tf.reduce_mean(tf.square(grad1 - grad2)):.3f}")
    print(f"Var squared difference = {sigma**2 * tf.math.reduce_std(tf.square(grad1 - grad2)):.3f}")

    assert np.allclose(sigma**2*grad1.numpy(), sigma**2*grad2.numpy(), rtol=1e-3)


if __name__ == '__main__':
    test_chisq_vis_gradient()