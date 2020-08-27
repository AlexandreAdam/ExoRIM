import tensorflow as tf
import numpy as np
from ExoRIM.definitions import dtype, mycomplex, rad2mas, triangle_pulse_f, mas2rad, TWOPI
from ExoRIM.base import PhysicalModelBase, BaselinesBase
from ExoRIM.operators import Baselines, closure_phase_operator, NDFTM, closure_phase_covariance_inverse, closure_fourier_matrices
import ExoRIM.log_likelihood as chisq
from scipy.linalg import pinv as pseudo_inverse, null_space


class PhysicalModelv1:
    """
    This model create discrete Fourier matrices and scales poorly on large number of pixels and baselines.
    """

    def __init__(self, pixels, mask_coordinates, chisq_term,
                 wavelength=0.5e-6, SNR=100, vis_phase_std=1e-5, logim=True, auto=False,
                 *args, **kwargs):
        """

        :param pixels: Number of pixels on the side of the reconstructed image
        :param SNR: Signal to noise ratio for a measurement
        """
        # super(PhysicalModelv1, self).__init__(*args, **kwargs)
        assert chisq_term in chisq.chi_squared.keys(), f"Available terms are {chisq.chi_squared.keys()}"
        self.chisq_term = chisq.chi_squared[chisq_term]
        self._chisq_term = chisq_term
        if auto:
            self.chisq_grad_term = chisq.chisq_gradients[chisq.chi_map[chisq_term]["Auto"]]
        else:
            self.chisq_grad_term = chisq.chisq_gradients[chisq.chi_map[chisq_term]["Analytical"]]

        self.pixels = pixels
        self.baselines = Baselines(mask_coordinates=mask_coordinates)
        self.resolution = rad2mas(wavelength / np.max(np.sqrt(self.baselines.UVC[:, 0]**2 + self.baselines.UVC[:, 1]**2)))

        self.plate_scale = self.compute_plate_scale(self.baselines, pixels, wavelength)
        self.smallest_scale = self.minimum_scale(self.baselines, self.plate_scale, wavelength)

        self.pulse = triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 0] / wavelength, mas2rad(self.resolution))
        self.pulse *= triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 1] / wavelength, mas2rad(self.resolution))

        self.CPO = closure_phase_operator(self.baselines)

        self.A = NDFTM(self.baselines.UVC, wavelength, pixels, self.plate_scale)
        self.p = self.A.shape[0]  # number of visibility samples
        self.q = self.CPO.shape[0]  # number of closure phases
        self.SNR = SNR
        self.sigma = tf.constant(1/self.SNR, dtype)
        self.phase_std = tf.constant(vis_phase_std, dtype)
        self.SIGMA = tf.constant(closure_phase_covariance_inverse(self.CPO, 1/SNR), dtype)
        A1, A2, A3 = closure_fourier_matrices(self.A, self.CPO)
        self.CPO = tf.constant(self.CPO, dtype)
        self.A = tf.constant(self.A, mycomplex)
        self.A1 = tf.constant(A1, mycomplex)
        self.A2 = tf.constant(A2, mycomplex)
        self.A3 = tf.constant(A3, mycomplex)
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.logim = logim # whether we reconstruct in tje log space or brightness space

    @tf.function
    def log_likelihood(self, image, X):
        """
        Compute the negative of the log likelihood of the image given interferometric data X.
        """
        return self.chisq_term(image, X, self)

    @tf.function
    def grad_log_likelihood(self, image, X):
        """
        Compute the chi squared gradient relative to image pixels given interferometric data X
        """
        return self.chisq_grad_term(image, X, self)

    def forward(self, image):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size)
        :return: A concatenation of complex visibilities and bispectra (dtype: tf.complex128)
        """
        X = self.fourier_transform(image)
        X = chisq.chisq_x_transformation[self._chisq_term](X, self)
        return X

    @tf.function
    def fourier_transform(self, image):
        im = tf.cast(image, mycomplex)
        flat = self.flatten(im)
        return tf.einsum("ij, ...j->...i", self.A, flat)  # tensordot broadcasted on batch_size

    @tf.function
    def bispectrum(self, image):
        im = tf.cast(image, mycomplex)
        flat = self.flatten(im)
        V1 = tf.einsum("ij, ...j -> ...i", self.A1, flat)
        V2 = tf.einsum("ij, ...j -> ...i", self.A2, flat)
        V3 = tf.einsum("ij, ...j -> ...i", self.A3, flat)
        return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other methods

    def simulate_noisy_data(self, images):
        batch = images.shape[0]
        X = self.fourier_transform(images)
        gain = self._aperture_gain(batch)
        phase_error = self._visibility_phase_noise(batch)
        amp = tf.cast(gain * tf.math.abs(X), mycomplex)
        phase = 1j * tf.cast(tf.math.angle(X) + phase_error, mycomplex)
        noisy_vis = amp * tf.math.exp(phase)
        noisy_X = chisq.chisq_x_transformation[self._chisq_term](noisy_vis, self)  # TODO make a better noise model
        return noisy_X

    def _visibility_phase_noise(self, batch):
        """
        Noise drawn from uniform distribution sort of simulate atmosphere disturbance in optical regime
        :param variance: Variance of the gaussian distribution in RAD
        """
        noise = np.random.normal(0, self.phase_std, size=[batch, self.baselines.nbap])
        visibility_phase_noise = np.einsum("ij, ...j -> ...i", self.baselines.BLM, noise)
        return tf.constant(visibility_phase_noise, dtype=dtype)

    def _aperture_gain(self, batch):
        """
        Simulate aperture gain based on signal to noise ratio. Since we suppose noise as zero mean, then SNR is
        S^2 / Var(Amplitude Noise).
        Gain is a normal distributed variable around 1. with standard deviation 1/sqrt(SNR)
        """
        return tf.constant(np.random.normal(1, 1/self.SNR, size=[batch, self.p]), dtype=dtype)

    @staticmethod
    def compute_plate_scale(B: Baselines, pixels, wavel) -> float:
        """
        Given a mask (with coordinate in meters) and a square image with *pixels* side coordinates, we evaluate the
        resolution of the virtual telescope and estimate a plate scale that will produce an image where object have
        frequency of the order that can be interpolated by the uv coverage.
        """
        rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        theta = rad2mas(1/rho)  # angular scale covered in mas
        # this picks out the bulk of the baseline frequencies, leaving out poorly constrained lower frequencies
        plate_scale = (np.median(theta) + 2*np.std(theta))/pixels
        return plate_scale

    @staticmethod
    def minimum_scale(B: Baselines, plate_scale, wavel) -> float:
        """
        Return the minimum scale that can be put into an image. Will constrained the lower bound
        of the pixel frequency.

        plate_scale: in mas/pixel

        return smallest scale in pixels
        """
        highest_frequency = np.max(np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2))/wavel  #1/RAD
        smallest_scale = rad2mas(1/highest_frequency) / plate_scale
        return smallest_scale




#TODO decide on convention for constructing bispectrum (either A2 is taken to be conjugate always or conjugate is
# explicit in every chi squared terms)

class MyopicPhysicalModel(PhysicalModelBase):
    """
    Physical Model with a pseudo amplitude and phase data model.
    """

    def __init__(self, pixels, mask_coordinates, chisq_term,
                 wavelength=0.5e-6, SNR=100, vis_phase_std=1e-5, logim=True, auto=False,
                 *args, **kwargs):
        """

        :param pixels: Number of pixels on the side of the reconstructed image
        :param SNR: Signal to noise ratio for a measurement
        """
        # super(PhysicalModelv1, self).__init__(*args, **kwargs)
        assert chisq_term in chisq.chi_squared.keys(), f"Available terms are {chisq.chi_squared.keys()}"
        self.chisq_term = chisq.chi_squared[chisq_term]
        self._chisq_term = chisq_term
        if auto:
            self.chisq_grad_term = chisq.chisq_gradients[chisq.chi_map[chisq_term]["Auto"]]
        else:
            self.chisq_grad_term = chisq.chisq_gradients[chisq.chi_map[chisq_term]["Analytical"]]
        self.pixels = pixels
        self.baselines = Baselines(mask_coordinates=mask_coordinates)
        self.resolution = rad2mas(
            wavelength / np.max(np.sqrt(self.baselines.UVC[:, 0] ** 2 + self.baselines.UVC[:, 1] ** 2)))

        self.plate_scale = self.compute_plate_scale(self.baselines, pixels, wavelength)
        self.smallest_scale = self.minimum_scale(self.baselines, self.plate_scale, wavelength)

        self.pulse = triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 0] / wavelength, mas2rad(self.plate_scale))
        self.pulse *= triangle_pulse_f(2 * np.pi * self.baselines.UVC[:, 1] / wavelength, mas2rad(self.plate_scale))

        self.CPO = closure_phase_operator(self.baselines)
        self.CPO_right_pseudo_inverse = pseudo_inverse(self.CPO)
        self.Bbar = self.baselines.BLM[:, 1:]
        # self.Bbar = null_space(self.CPO)

        self.A = NDFTM(self.baselines.UVC, wavelength, pixels, self.plate_scale)
        self.p = self.A.shape[0]  # number of visibility samples
        self.q = self.CPO.shape[0]  # number of closure phases
        self.SNR = SNR
        self.sigma = tf.constant(1 / self.SNR, dtype)
        self.phase_std = tf.constant(vis_phase_std, dtype)
        self.SIGMA = tf.constant(closure_phase_covariance_inverse(self.CPO, 1 / SNR), dtype)
        A1, A2, A3 = closure_fourier_matrices(self.A, self.CPO)
        self.CPO = tf.constant(self.CPO, dtype)
        self.CPO_right_pseudo_inverse = tf.constant(self.CPO_right_pseudo_inverse, dtype)
        self.Bbar = tf.constant(self.Bbar, dtype)
        self.A = tf.constant(self.A, mycomplex)
        self.A1 = tf.constant(A1, mycomplex)
        self.A2 = tf.constant(A2, mycomplex)
        self.A3 = tf.constant(A3, mycomplex)
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.logim = logim  # whether we reconstruct in tje log space or brightness space

    @tf.function
    def log_likelihood(self, image, X):
        """
        Compute the negative of the log likelihood of the image given interferometric data X.
        """
        return self.chisq_term(image, X, self)

    @tf.function
    def grad_log_likelihood(self, image, X):
        """
        Compute the chi squared gradient relative to image pixels given interferometric data X
        """
        return self.chisq_grad_term(image, X, self)

    @tf.function
    def forward(self, image):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size)
        :return: A concatenation of complex visibilities and bispectra (dtype: tf.complex128)
        """
        X = self.fourier_transform(image)
        X = chisq.chisq_x_transformation[self._chisq_term](X, self)
        return X

    @tf.function
    def fourier_transform(self, image):
        im = tf.cast(image, mycomplex)
        flat = self.flatten(im)
        return tf.einsum("ij, ...j->...i", self.A, flat)  # tensordot broadcasted on batch_size

    @tf.function
    def bispectrum(self, image):
        im = tf.cast(image, mycomplex)
        flat = self.flatten(im)
        V1 = tf.einsum("ij, ...j -> ...i", self.A1, flat)
        V2 = tf.einsum("ij, ...j -> ...i", self.A2, flat)
        V3 = tf.einsum("ij, ...j -> ...i", self.A3, flat)
        return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other methods

    def simulate_noisy_data(self, images):
        batch = images.shape[0]
        X = self.fourier_transform(images)
        gain = self._aperture_gain(batch)
        phase_error = self._visibility_phase_noise(batch)
        amp = tf.cast(gain * tf.math.abs(X), mycomplex)
        phase = 1j * tf.cast(tf.math.angle(X) + phase_error, mycomplex)
        noisy_vis = amp * tf.math.exp(phase)
        noisy_X = chisq.chisq_x_transformation[self._chisq_term](noisy_vis, self)  # TODO make a better noise model
        return noisy_X

    def _visibility_phase_noise(self, batch):
        """
        Noise drawn from uniform distribution sort of simulate atmosphere disturbance in optical regime
        :param variance: Variance of the gaussian distribution in RAD
        """
        noise = np.random.normal(0, self.phase_std, size=[batch, self.baselines.nbap])
        visibility_phase_noise = np.einsum("ij, ...j -> ...i", self.baselines.BLM, noise)
        return tf.constant(visibility_phase_noise, dtype=dtype)

    def _aperture_gain(self, batch):
        """
        Simulate aperture gain based on signal to noise ratio. Since we suppose noise as zero mean, then SNR is
        S^2 / Var(Amplitude Noise).
        Gain is a normal distributed variable around 1. with standard deviation 1/sqrt(SNR)
        """
        return tf.constant(np.random.normal(1, 1/self.SNR, size=[batch, self.p]), dtype=dtype)

    @staticmethod
    def compute_plate_scale(B: Baselines, pixels, wavel) -> float:
        """
        Given a mask (with coordinate in meters) and a square image with *pixels* side coordinates, we evaluate the
        resolution of the virtual telescope and estimate a plate scale that will produce an image where object have
        frequency of the order that can be interpolated by the uv coverage.
        """
        rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
        theta = rad2mas(1/rho)  # angular scale covered in mas
        # this picks out the bulk of the baseline frequencies, leaving out poorly constrained lower frequencies
        plate_scale = (np.median(theta) + 2*np.std(theta))/pixels
        return plate_scale

    @staticmethod
    def minimum_scale(B: Baselines, plate_scale, wavel) -> float:
        """
        Return the minimum scale that can be put into an image. Will constrained the lower bound
        of the pixel frequency.

        plate_scale: in mas/pixel

        return smallest scale in pixels
        """
        highest_frequency = np.max(np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2))/wavel  #1/RAD
        smallest_scale = rad2mas(1/highest_frequency) / plate_scale
        return smallest_scale




#TODO make a physical model with random uv coverage
# This will 100% necessitate NFFT!
# So create grad equations in definitions that support adjoint NFFT transform instead of taking the matrix
class PhysicalModelv2:
    """
    Since computing the Discrete Fourier Matrices for each sample requires a large amount of both memory and
    computation, this version enables a Dataset generator to precompute the transformation plan for a batch of
    image/uv coverage pair.

    This class works with the pynufft which should be installed by the user from
    https://github.com/jyhmiinlin/pynufft using git.
    """
    def __init__(self, pixels, mask_coordinates, wavelength, plate_scale, SNR, vis_phase_std=np.pi/3):
        self.version = "v2"
        self.wavelength = wavelength
        self.plate_scale = plate_scale
        self.SNR = SNR
        self.phase_std = vis_phase_std # TODO make a noise model for this
        self.pixels = pixels
        # TODO add an observation function to baseline to simulate rotation of the earth for mm/sub mm observations.


PhysicalModel = PhysicalModelv1
