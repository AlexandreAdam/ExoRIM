from exorim.operators import Operators
from exorim.definitions import rad2mas, DTYPE, MYCOMPLEX, cast_to_complex_flatten, LOG10, LOGFLOOR
import exorim.inference as chisq
import numpy as np
import tensorflow as tf

GOLAY9 = tf.constant(np.array([
 [0.00000,  2.30000],
 [0.00000,  3.45000],
 [1.99186, -1.15000],
 [2.98779, -1.72500],
 [-1.99186, -1.15000],
 [-2.98779, -1.72500],
 [-1.99186,  2.30000],
 [-0.99593, -2.98779],
 [2.98779,  0.49796],
]), dtype=DTYPE)

JWST_NIRISS_MASK = tf.constant(np.array([ #V2/m and V3/m coordinates
    [ 1.143,  1.980],  # C1
    [ 2.282,  1.317],  # B2
    [ 2.286,  0.000],  # C2
    [ 0.000, -2.635],  # B4
    [-2.282, -1.317],  # B5
    [-2.282,  1.317],  # B6
    [-1.143,  1.980]   # C6
]), dtype=DTYPE)


class PhysicalModel:
    def __init__(
            self,
            pixels,
            mask_coordinates=JWST_NIRISS_MASK,
            wavelength=3.8e-6,
            chi_squared="append_visibility_amplitude_closure_phase",
            logim=True,
            redundant=False,  # closure phases
            oversampling_factor=None,
            plate_scale=None,  # in mas units
            beta=1.
    ):
        assert chi_squared in ["append_visibility_amplitude_closure_phase", "visibility", "visibility_amplitude"]
        self._chi_squared = chi_squared
        self.pixels = pixels
        self.operators = Operators(mask_coordinates=mask_coordinates, wavelength=wavelength, redundant=redundant)
        self.CPO = tf.constant(self.operators.CPO, DTYPE)
        if plate_scale is None:
            self.plate_scale = self.compute_plate_scale(wavelength, oversampling_factor)
        else:
            self.plate_scale = plate_scale
        A, Ainv, A1, A2, A3 = self.operators.build_operators(pixels, self.plate_scale)
        V1, V2, V3 = self.operators.closure_baseline_projectors()
        self.A = tf.constant(A, MYCOMPLEX)
        self.Ainv = tf.constant(Ainv, MYCOMPLEX)
        self.A1 = tf.constant(A1, MYCOMPLEX)
        self.A2 = tf.constant(A2, MYCOMPLEX)
        self.A3 = tf.constant(A3, MYCOMPLEX)
        self.V1 = tf.constant(V1, MYCOMPLEX)
        self.V2 = tf.constant(V2, MYCOMPLEX)
        self.V3 = tf.constant(V3, MYCOMPLEX)
        self.logim = logim
        self.nbuv = self.operators.nbuv
        self.nbcp = self.operators.CPO.shape[0]
        self.beta = beta

        if self.logim:
            self.image_link = lambda xi: 10**xi  # xi to image
            self.image_inverse_link = lambda image: tf.math.log(tf.maximum(image, LOGFLOOR)) / LOG10  # image to xi
            self.gradient_link = lambda image, grad: image * grad * LOG10
        else:
            self.image_link = lambda image: image
            self.image_inverse_link = lambda image: image
            self.gradient_link = lambda image, grad: grad

    def grad_chi_squared(self, xi, X, sigma):
        image = self.image_link(xi)
        grad, chi_squared = chisq.chisq_gradients[self._chi_squared](image=image, X=X, phys=self, sigma=sigma, beta=self.beta)
        return self.gradient_link(image, grad), chi_squared

    def chi_squared(self, xi, X, sigma):
        image = self.image_link(xi)
        return chisq.chi_squared[self._chi_squared](image=image, X=X, phys=self, sigma=sigma)

    def forward(self, image):
        V = tf.einsum("ij, ...j->...i", self.A, cast_to_complex_flatten(image))
        X = chisq.v_transformation[self._chi_squared](V, self)
        return X

    def inverse(self, X):
        batch_size = X.shape[0]
        V = X[..., :self.nbuv]
        im = tf.math.abs(tf.einsum("ij, ...j->...i", self.Ainv, tf.cast(V, MYCOMPLEX)))
        im = tf.reshape(im, shape=[batch_size, self.pixels, self.pixels, 1])
        return im

    def noisy_forward(self, image, sigma):
        batch = image.shape[0]
        V = tf.einsum("ij, ...j->...i", self.A, cast_to_complex_flatten(image))
        epsilon = self.circularly_gaussian_random_variable(batch, sigma)
        noisy_V = V + epsilon
        X, sigma = chisq.v_sigma_transformation[self._chi_squared](noisy_V, self, sigma)
        return X, sigma

    def bispectrum(self, image):
        flat = cast_to_complex_flatten(image)
        V1 = tf.einsum("ij, ...j -> ...i", self.A1, flat)
        V2 = tf.einsum("ij, ...j -> ...i", self.A2, flat)
        V3 = tf.einsum("ij, ...j -> ...i", self.A3, flat)
        return V1 * tf.math.conj(V2) * V3

    def visibility(self, image):
        return tf.einsum("ij, ...j -> ...i", self.A, cast_to_complex_flatten(image))

    def circularly_gaussian_random_variable(self, batch_size, sigma):
        z = tf.cast(tf.complex(
            real=tf.random.normal(shape=[batch_size, self.nbuv], stddev=sigma),
            imag=tf.random.normal(shape=[batch_size, self.nbuv], stddev=sigma)
        ), MYCOMPLEX)
        return z

    def compute_plate_scale(self, wavel, oversampling_factor=None) -> float:
        """ Compute the angular size of a pixel """
        rho = np.sqrt(self.operators.UVC[:, 0]**2 + self.operators.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        fov = rad2mas(1/rho).max()
        resolution = (rad2mas(1/2/rho)).min()  # Michelson criterion = lambda / 2b radians
        if oversampling_factor is None:
            oversampling_factor = self.pixels * resolution / fov
        plate_scale = resolution / oversampling_factor
        return plate_scale


if __name__ == '__main__':
    from exorim.datasets import NCompanions
    from exorim import PhysicalModel
    import matplotlib.pyplot as plt
    from matplotlib.colors import CenteredNorm
    from exorim import PhysicalModel

    phys = PhysicalModel(128, oversampling_factor=None, logim=False, plate_scale=5)
    extent = [*[-phys.plate_scale * phys.pixels, phys.plate_scale * phys.pixels]*2]
    D = NCompanions(phys, 1, 1, width=2)
    X, y, sigma = next(D)
    x_true = phys.forward(y)
    plt.figure()
    plt.title("True image")
    plt.imshow(y[0, ..., 0], cmap="hot", extent=extent)
    plt.colorbar()

    y_pred = phys.inverse(X)
    # _, y_pred, _ = next(D)

    plt.figure()
    plt.title("Pred image")
    plt.imshow(y_pred[0, ..., 0], cmap="hot")
    plt.colorbar()

    # Tensorflow backprop gradient wrt model parameters
    self = phys
    xi = self.image_inverse_link(y_pred)
    with tf.GradientTape() as tape:
        tape.watch(xi)
        image = self.image_link(xi)
        x_pred = self.forward(image)
        chi_squared = tf.reduce_sum((x_pred[..., :self.nbuv] - X[..., :self.nbuv]) ** 2 / sigma[:, :self.nbuv]**2, axis=1)
        chi_squared += 2 * tf.reduce_sum((1 - tf.cos(x_pred[..., self.nbuv:] - X[..., self.nbuv:])) / sigma[..., self.nbuv:]**2, axis=1)
        cost = tf.reduce_mean(chi_squared)
    grad = tape.gradient(cost, xi)

    print(chi_squared)
    plt.figure()
    plt.title("Autodiff grad")
    plt.imshow(grad[0, ..., 0], cmap="seismic", norm=CenteredNorm())
    plt.colorbar()
    # plt.show()

    grad, chi_squared = chisq.chisq_gradients[self._chi_squared](image=self.image_link(xi), X=X, phys=self, sigma=sigma)
    grad = self.gradient_link(self.image_link(xi), grad)

    print(chi_squared)
    plt.figure()
    plt.title("Analytical grad")
    plt.imshow(grad[0, ..., 0], cmap="seismic", norm=CenteredNorm())
    plt.colorbar()
    plt.show()
