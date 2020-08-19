from ExoRIM.operators import Baselines, NDFTM
from ExoRIM.log_likelihood import *
import time
import numpy as np


def test_chisq_vis_gradient():
    pixels = 32
    mask = np.random.normal(0, 6, [21, 2])
    B = Baselines(mask)
    A = tf.constant(NDFTM(B.UVC, 0.5e-6, pixels, 0.3), mycomplex)
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
    noise = tf.constant(noise.reshape((1, pixels, pixels)), dtype)

    vis = tf.constant(np.dot(A, image.flatten()).reshape((1, -1)), mycomplex)

    start = time.time()
    grad1 = chisq_gradient_complex_visibility_auto(noise, A, vis, sigma)
    print(f"Auto grad took {time.time() - start:.3f} sec")
    start = time.time()
    grad2 = chisq_gradient_complex_visibility_analytic(noise, A, vis, sigma)
    print(f"Analytical grad took {time.time() - start:.3f} sec")
    assert np.allclose(sigma**2*grad1.numpy(), sigma**2*grad2.numpy(), rtol=1e-3)

    # TODO investigate how to make AutoGrad match analytical one --> simulate the conjugate of A somehow.


if __name__ == '__main__':
    test_chisq_vis_gradient()