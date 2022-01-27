from .operators import Operators
from .definitions import rad2mas, DTYPE, MYCOMPLEX, cast_to_complex_flatten, LOG10
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
            oversampling_factor=None,
            logim=True
    ):
        assert chi_squared in ["append_visibility_amplitude_closure_phase", "visibility", "visibility_amplitude"]
        self._chi_squared = chi_squared
        self.pixels = pixels
        self.operators = Operators(mask_coordinates=mask_coordinates, wavelength=wavelength)
        self.CPO = tf.constant(self.operators.CPO, DTYPE)
        self.plate_scale = self.compute_plate_scale(wavelength, oversampling_factor)
        A, A1, A2, A3 = self.operators.build_operators(pixels, self.plate_scale)
        V1, V2, V3 = self.operators.closure_baseline_projectors()
        self.A = tf.constant(A, MYCOMPLEX)
        self.A1 = tf.constant(A1, MYCOMPLEX)
        self.A2 = tf.constant(A2, MYCOMPLEX)
        self.A3 = tf.constant(A3, MYCOMPLEX)
        self.V1 = tf.constant(V1, MYCOMPLEX)
        self.V2 = tf.constant(V2, MYCOMPLEX)
        self.V3 = tf.constant(V3, MYCOMPLEX)
        self.logim = logim
        self.nbuv = self.operators.nbuv

        if self.logim:
            self.image_link = lambda image: 10**image
            self.gradient_link = lambda image, grad: image * grad * LOG10
        else:
            self.image_link = lambda image: image
            self.gradient_link = lambda image, grad: grad

    def grad_log_likelihood(self, image, X, sigma):
        image = self.image_link(image)
        grad = chisq.chisq_gradients[self._chi_squared](image=image, X=X, phys=self, sigma=sigma)
        return self.gradient_link(image, grad)

    def forward(self, image):
        V = tf.einsum("ij, ...j->...i", self.A, cast_to_complex_flatten(image))
        X = chisq.v_transformation[self._chi_squared](V, self)
        return X

    def noisy_forward(self, image, amplitude_noise_std, phase_noise_std):
        batch = image.shape[0]
        V = tf.einsum("ij, ...j->...i", self.A, cast_to_complex_flatten(image))
        gain = self.gain(batch, amplitude_noise_std, phase_noise_std) # TODO find principled way to add noise and infer closure phase noise from it
        noisy_V = gain * V
        X = chisq.v_transformation[self._chi_squared](noisy_V, self)
        return X

    def chi_squared(self, image, X, sigma):
        return chisq.chi_squared[self._chi_squared](image=image, X=X, phys=self, sigma=sigma)

    def bispectrum(self, image):
        flat = cast_to_complex_flatten(image)
        V1 = tf.einsum("ij, ...j -> ...i", self.A1, flat)
        V2 = tf.einsum("ij, ...j -> ...i", self.A2, flat)
        V3 = tf.einsum("ij, ...j -> ...i", self.A3, flat)
        return V1 * tf.math.conj(V2) * V3

    def visibility(self, image):
        return tf.einsum("ij, ...j -> ...i", self.A, cast_to_complex_flatten(image))

    def gain(self, batch, amplitude_noise_std, phase_noise_std):
        amp = np.random.normal(1, amplitude_noise_std, size=[batch, self.nbuv])
        phase = np.exp(1j * np.random.normal(0, phase_noise_std, size=[batch, self.nbuv]))
        return tf.constant(amp * phase, dtype=MYCOMPLEX)

    def compute_plate_scale(self, wavel, oversampling_factor=None) -> float:
        """ Compute the angular size of a pixel """
        rho = np.sqrt(self.operators.UVC[:, 0]**2 + self.operators.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        fov = rad2mas(1/rho).max()
        resolution = (rad2mas(1/2/rho)).min()  # Michelson criterion = lambda / 2b radians
        if oversampling_factor is None:
            oversampling_factor = self.pixels * resolution / fov
        plate_scale = resolution / oversampling_factor
        return plate_scale
