from exorim.interferometry.operators import NDFTM, \
    closure_fourier_matrices, Baselines, \
    closure_phase_operator, closure_phase_covariance_inverse
from exorim.definitions import rad2mas, DTYPE, MYCOMPLEX
import exorim.inference.log_likelihood as chisq
from exorim.inference.regularizers import entropy
import numpy as np
import tensorflow as tf

#TODO figure out how to take into account rotation
JWST_NIRISS_MASK = tf.constant(np.array([ #V2/m and V3/m coordinates
    [ 1.143,  1.980],  # C1
    [ 2.282,  1.317],  # B2
    [ 2.286,  0.000],  # C2
    [ 0.000, -2.635],  # B4
    [-2.282, -1.317],  # B5
    [-2.282,  1.317],  # B6
    [-1.143,  1.980]   # C6
]), dtype=DTYPE)
# don't forget to invert y axis!
JWST_NIRISS_MASK[:, 1] = -JWST_NIRISS_MASK[:, 1]

# f380m_pistons_amplitude = tf.constant(
#     np.array([0.00079326, 0.0003999, 0.00025881, 0.00084522, 0.00018091,
#            0.00022733, 0.000813, 0.00076525, 0.00054603, 0.00050279,
#            0.00062676, 0.00063966, 0.00025679, 0.0002882, 0.00035917,
#            0.00109878, 0.00022232, 0.00036921, 0.00026031, 0.00037381,
#            0.00045959]),
#     dtype=DTYPE)
#
#


f380m_WAVELENGTH = 3.826025395053875e-06


class PhysicalModel:
    def __init__(self,
                 pixels,
                 mask_coordinates=JWST_NIRISS_MASK,
                 loglikelihood="append_visibility_amplitude_closure_phase",
                 analytic=False,
                 temperature=1,
                 wavelength=f380m_WAVELENGTH,
                 SNR=100, # from observation, to be added in quadrature with pistons uncertainties
                 logim=True):
        assert loglikelihood in ["append_visibility_amplitude_closure_phase", "visibilities", "visibility_amplitude"]
        self.temperature = tf.constant(temperature, dtype=DTYPE)
        self._loglikelihood = loglikelihood
        self._analytic = analytic
        self.pixels = pixels
        self.baselines = Baselines(mask_coordinates=mask_coordinates)
        self.resolution = rad2mas(wavelength / np.max(np.sqrt(self.baselines.UVC[:, 0]**2 + self.baselines.UVC[:, 1]**2)))

        self.plate_scale = self.compute_plate_scale(self.baselines, wavelength)

        #TODO figure out how to use these
        # self.pulse = triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 0] / wavelength, mas2rad(self.resolution))
        # self.pulse *= triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 1] / wavelength, mas2rad(self.resolution))

        self.CPO = closure_phase_operator(self.baselines)

        self.A = NDFTM(self.baselines.UVC, wavelength, pixels, self.plate_scale)
        self.N = mask_coordinates.shape[0]  # number of apertures
        self.p = self.A.shape[0]  # number of visibility samples
        self.q = self.CPO.shape[0]  # number of closure phases
        self.SNR = SNR
        self.sigma = tf.constant(np.sqrt((1 / self.SNR)**2 + f380m_pistons_amplitude**2), DTYPE)
        self.phase_std = f380m_pistons_angles
        A1, A2, A3 = closure_fourier_matrices(self.A, self.CPO)
        self.CPO = tf.constant(self.CPO, DTYPE)
        self.A = tf.constant(self.A, MYCOMPLEX)
        self.A1 = tf.constant(A1, MYCOMPLEX)
        self.A2 = tf.constant(A2, MYCOMPLEX)
        self.A3 = tf.constant(A3, MYCOMPLEX)
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.logim = logim # whether we reconstruct in tje log space or brightness space

    def grad_log_likelihood(self, image, X, append=False):
        """
        Compute the chi squared gradient relative to image pixels given interferometric data X
        """
        if self._loglikelihood == "append_visibility_amplitude_closure_phase":
            amp = X[..., :self.p]
            cp = X[..., self.p:]
            if self._analytic:
                if self.logim:
                    image = tf.math.exp(image)
                    grad = chisq.chisq_gradient_amplitude(image, amp, self)
                    grad += chisq.chisq_gradient_closure_phasor(image, cp, self)
                    grad *= image
                    return grad / self.temperature
                else:
                    grad = chisq.chisq_gradient_amplitude(image, amp, self)
                    grad += chisq.chisq_gradient_closure_phasor(image, cp, self)
                    return grad / self.temperature
            # With autograd
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(image)
                    if self.logim:
                        _image = tf.math.exp(image)
                    else:
                        _image = image
                    ll = chisq.chi_squared_amplitude(_image, amp, self)
                    ll += chisq.chi_squared_closure_phasor(_image, cp, self)
                    ll /= self.temperature
                grad = tape.gradient(ll, image)
                return grad
        else:  # case where we don't append two terms, so visibilities and amplitude alone go here
            if self._analytic:
                if self.logim:
                    image = np.exp(image)
                    return image * chisq.chisq_gradients[self._loglikelihood](image, X, self) / self.temperature
                else:
                    return chisq.chisq_gradients[self._loglikelihood](image, X, self) / self.temperature
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(image)
                    if self.logim:
                        _image = tf.math.exp(image)
                    else:
                        _image = image
                    ll = chisq.chi_squared[self._loglikelihood](_image, X, self)
                    ll /= self.temperature
                grad = tape.gradient(ll, image)
                return grad

    def forward(self, image):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size)
        :return: A concatenation of complex visibilities and bispectra (dtype: tf.complex128)
        """
        V = self.fourier_transform(image)
        X = chisq.chisq_x_transformation[self._loglikelihood](V, self)
        return X

    def chi_squared(self, image, X):
        if self.logim:
            image = np.exp(image)
        if self._loglikelihood == "append_visibility_amplitude_closure_phase":
            amp = X[..., :self.p]
            cp  = X[..., self.p:]
            ll = chisq.chi_squared_amplitude(image, amp, self)
            ll += chisq.chi_squared_closure_phasor(image, cp, self)
            ll /= self.temperature
        else:
            ll = chisq.chi_squared[self._loglikelihood](image, X, self)
            ll /= self.temperature
        return ll

    def fourier_transform(self, image):
        im = tf.cast(image, MYCOMPLEX)
        flat = self.flatten(im)
        return tf.einsum("ij, ...j->...i", self.A, flat)  # tensordot broadcasted on batch_size

    def bispectrum(self, image):
        im = tf.cast(image, MYCOMPLEX)
        flat = self.flatten(im)
        V1 = tf.einsum("ij, ...j -> ...i", self.A1, flat)
        V2 = tf.einsum("ij, ...j -> ...i", self.A2, flat)
        V3 = tf.einsum("ij, ...j -> ...i", self.A3, flat)
        return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other methods

    def noisy_forward(self, images):
        batch = images.shape[0]
        V = self.fourier_transform(images)
        gain = self._gain(batch)
        noisy_V = gain * V
        X = chisq.chisq_x_transformation[self._loglikelihood](noisy_V, self)
        return X

    def _gain(self, batch):
        amp = np.random.normal(1, self.sigma, size=[batch, self.p])
        phase = np.exp(1j * np.random.normal(0, self.phase_std, size=[batch, self.p]))  # baseline intrinsic phase error
        return tf.constant(amp * phase, dtype=MYCOMPLEX)

    def compute_plate_scale(self, B: Baselines, wavel) -> float:
        # by default, use FOV/pixels and hope this satisfy Nyquist sampling criterion
        rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        fov = rad2mas(1/rho).max()
        B = (1/rad2mas(1/rho)).max() # largest frequency in the signal in mas^{-1}
        plate_scale = fov / self.pixels
        if 1/plate_scale <= 2 * B:
            print("Nyquist sampling criterion is not satisfied")
        return plate_scale
