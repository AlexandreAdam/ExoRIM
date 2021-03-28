from exorim.interferometry.operators import NDFTM, \
    closure_fourier_matrices, Baselines, \
    closure_phase_operator, closure_phase_covariance_inverse
from exorim.definitions import rad2mas, DTYPE, MYCOMPLEX
import exorim.inference.log_likelihood as chisq
from exorim.inference.regularizers import entropy
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


class PhysicalModel:
    def __init__(self,
                 pixels,
                 mask_coordinates=GOLAY9,
                 loglikelihood="append_visibility_amplitude_closure_phase",
                 analytic=False,
                 temperature=1,
                 wavelength=0.5e-6,
                 SNR=100,
                 vis_phase_std=0.1,
                 logim=True,
                 lam=0 # regularization
            ):
        assert loglikelihood in ["append_visibility_amplitude_closure_phase", "visibilities", "visibility_amplitude"]
        self.temperature = tf.constant(temperature, dtype=DTYPE)
        self._loglikelihood = loglikelihood
        self._analytic = analytic
        self.pixels = pixels
        self.baselines = Baselines(mask_coordinates=mask_coordinates)
        self.resolution = rad2mas(wavelength / np.max(np.sqrt(self.baselines.UVC[:, 0]**2 + self.baselines.UVC[:, 1]**2)))

        self.plate_scale = self.compute_plate_scale(self.baselines, pixels, wavelength)

        #TODO figure out how to use these
        # self.pulse = triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 0] / wavelength, mas2rad(self.resolution))
        # self.pulse *= triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 1] / wavelength, mas2rad(self.resolution))

        self.CPO = closure_phase_operator(self.baselines)

        self.A = NDFTM(self.baselines.UVC, wavelength, pixels, self.plate_scale)
        self.N = mask_coordinates.shape[0]  # number of apertures
        self.p = self.A.shape[0]  # number of visibility samples
        self.q = self.CPO.shape[0]  # number of closure phases
        self.SNR = SNR
        self.sigma = tf.constant(1 / self.SNR, DTYPE)
        self.phase_std = tf.constant(vis_phase_std, DTYPE)
        self.SIGMA = tf.constant(closure_phase_covariance_inverse(self.CPO, 1/SNR), DTYPE)
        A1, A2, A3 = closure_fourier_matrices(self.A, self.CPO)
        self.CPO = tf.constant(self.CPO, DTYPE)
        self.A = tf.constant(self.A, MYCOMPLEX)
        self.A1 = tf.constant(A1, MYCOMPLEX)
        self.A2 = tf.constant(A2, MYCOMPLEX)
        self.A3 = tf.constant(A3, MYCOMPLEX)
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.logim = logim # whether we reconstruct in tje log space or brightness space

        # TODO find a better way to do this
        prior = np.zeros(shape=[1, pixels, pixels, 1])
        x = np.arange(pixels) - pixels//2 + 0.5
        xx, yy = np.meshgrid(x, x)
        rho = np.hypot(xx, yy)
        prior[0, ..., 0] += np.exp(-0.5 * rho**2/(pixels/4)**2)
        prior /= prior.sum()
        self.prior = tf.constant(prior, DTYPE)
        self.lam = lam

    def grad_log_likelihood(self, image, X, append=False):
        """
        Compute the chi squared gradient relative to image pixels given interferometric data X
        """
        #TODO put this gradient in loglikelihood file
        if self._loglikelihood == "append_visibility_amplitude_closure_phase":
            amp = X[..., :self.p]
            cp = X[..., self.p:]
            if self._analytic:
                if self.logim:
                    image = tf.math.exp(image)
                    grad = chisq.chisq_gradient_amplitude(image, amp, self)
                    grad += chisq.chisq_gradient_closure_phasor(image, cp, self) #TODO entropy analytical gradient?
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
                    ll += self.lam * entropy(_image, self.prior)
                    ll /= self.temperature
                grad = tape.gradient(ll, image)
                return grad
        else:  # case where we dont append two terms, so visibilities and amplitude alone go here
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

    def simulate_noisy_data(self, images):
        batch = images.shape[0]
        V = self.fourier_transform(images)
        gain = self._aperture_gain(batch)
        phase_error = self._visibility_phase_noise(batch)
        amp = tf.cast(gain * tf.math.abs(V), MYCOMPLEX)
        phase = 1j * tf.cast(tf.math.angle(V) + phase_error, MYCOMPLEX)
        noisy_V = amp * tf.math.exp(phase)
        X = chisq.chisq_x_transformation[self._loglikelihood](noisy_V, self)
        return X

    def _visibility_phase_noise(self, batch):
        """
        Noise drawn from uniform distribution sort of simulate atmosphere disturbance in optical regime
        :param variance: Variance of the gaussian distribution in RAD
        """
        noise = np.random.normal(0, self.phase_std, size=[batch, self.baselines.nbap])
        visibility_phase_noise = np.einsum("ij, ...j -> ...i", self.baselines.BLM, noise)
        return tf.constant(visibility_phase_noise, dtype=DTYPE)

    def _aperture_gain(self, batch):
        """
        Simulate aperture gain based on signal to noise ratio. Since we suppose noise as zero mean, then SNR is
        S^2 / Var(Amplitude Noise).
        Gain is a normal distributed variable around 1. with standard deviation 1/sqrt(SNR)
        """
        return tf.constant(np.random.normal(1, 1/self.SNR, size=[batch, self.p]), dtype=DTYPE)

    def compute_plate_scale(self, B: Baselines, pixels, wavel) -> float:
        # by default, use FOV/pixels and hope this satisfy Nyquist sampling criterion
        rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        fov = rad2mas(1/rho).max()
        B = (1/rad2mas(1/rho)).max() # largest frequency in the signal in mas^{-1}
        plate_scale = fov / self.pixels
        if 1/plate_scale <= 2 * B:
            print("Nyquist sampling criterion is not satisfied")
        return plate_scale
