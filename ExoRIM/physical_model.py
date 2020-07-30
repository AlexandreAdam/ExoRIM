import tensorflow as tf
import numpy as np
from ExoRIM.definitions import dtype, mycomplex, chisqgrad_vis, chisqgrad_bs, mas2rad, chisqgrad_amp, chisqgrad_cphase
from ExoRIM.operators import Baselines, phase_closure_operator, NDFTM


class PhysicalModel:
    """
    This model create discrete Fourier matrices and scales poorly on large number of pixels and baselines.
    """
    def __init__(self, pixels, mask_coordinates, wavelength, plate_scale, SNR, vis_phase_std=1e-5):
        """

        :param pixels: Number of pixels on the side of the reconstructed image
        :param SNR: Signal to noise ratio for a measurement
        """
        self.version = "v1"
        self.pixels = pixels
        self.baselines = Baselines(mask_coordinates=mask_coordinates)
        self.CPO = phase_closure_operator(self.baselines)
        self.A = NDFTM(self.baselines.UVC, wavelength, pixels, plate_scale)
        self.A_adjoint = NDFTM(self.baselines.UVC, wavelength, pixels, plate_scale, inv=True)
        self.p = self.A.shape[0]  # number of visibility samples
        self.q = self.CPO.shape[0]  # number of independent closure phases
        self.SNR = SNR
        self.phase_std = vis_phase_std
        # create matrices that project visibilities to bispectra (V1 = V_{ij}, V_2 = V_{jk} and V_3 = V_{ki})
        bisp_i = np.where(self.CPO != 0)
        V1_i = (bisp_i[0][0::3], bisp_i[1][0::3])
        V2_i = (bisp_i[0][1::3], bisp_i[1][1::3])
        V3_i = (bisp_i[0][2::3], bisp_i[1][2::3])
        self.V1_projector = np.zeros(shape=(self.q, self.p))
        self.V1_projector[V1_i] += 1.0
        self.V1_projector = tf.constant(self.V1_projector, dtype=mycomplex)
        self.V2_projector = np.zeros(shape=(self.q, self.p))
        self.V2_projector[V2_i] += 1.0
        self.V2_projector = tf.constant(self.V2_projector, dtype=mycomplex)
        self.V3_projector = np.zeros(shape=(self.q, self.p))
        self.V3_projector[V3_i] += 1.0
        self.V3_projector = tf.constant(self.V3_projector, dtype=mycomplex)
        self.CPO = tf.constant(self.CPO, dtype=dtype)
        self.A = tf.constant(self.A, dtype=mycomplex)
        self.A_adjoint = tf.constant(self.A_adjoint, dtype=mycomplex)
        # Discrete Fourier Transform Matrices
        self.A1 = tf.tensordot(self.V1_projector, self.A, axes=1)
        self.A2 = tf.tensordot(self.V2_projector, self.A, axes=1)
        self.A3 = tf.tensordot(self.V3_projector, self.A, axes=1)

    @tf.function
    def forward(self, image):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size)
        :return: A concatenation of complex visibilities and bispectra (dtype: tf.complex128)
        """
        visibilities = self.fourier_transform(image)
        bispectra = self.bispectrum(visibilities)
        y = tf.concat([visibilities, bispectra], axis=1)  # p + q length vectors of type complex128
        return y

    @tf.function
    def fourier_transform(self, image):
        im = tf.cast(image, mycomplex)
        flat = tf.keras.layers.Flatten(data_format="channels_last")(im)
        return tf.einsum("...ij, ...j->...i", self.A, flat)  # tensordot broadcasted on batch_size

    @tf.function
    def inverse_fourier_transform(self, X):
        amp = tf.cast(X[..., :self.p], mycomplex)
        flat = tf.einsum("...ij, ...j->...i", self.A_adjoint, amp)
        flat = tf.square(tf.math.abs(flat))  # intensity is square of amplitude
        flat = tf.cast(flat, dtype)
        return tf.reshape(flat, [-1, self.pixels, self.pixels, 1])

    @tf.function
    def bispectrum(self, V):
        V1 = tf.einsum("ij, ...j -> ...i", self.V1_projector, V)
        V2 = tf.einsum("ij, ...j -> ...i", self.V2_projector, V)
        V3 = tf.einsum("ij, ...j -> ...i", self.V3_projector, V)
        return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other methods

    @tf.function
    def log_likelihood_v1(self, Y_pred, X):
        """
        :param Y_pred: reconstructed image
        :param X: interferometric data from measurements
        """
        sigma_amp = tf.math.abs(X[..., :self.p]) / self.SNR
        sigma_cp = tf.cast(tf.math.sqrt(3 / self.SNR ** 2), dtype)
        X_pred = self.forward(Y_pred)
        chi2_amp = tf.reduce_mean(tf.square(tf.math.abs(X_pred[..., :self.p] - X[..., :self.p])/(sigma_amp + 1e-6)), axis=1)
        cp_pred = tf.math.angle(X_pred[..., self.p:])
        cp_true = tf.math.angle(X[..., self.p:])
        chi2_cp = tf.reduce_mean(tf.square((cp_pred - cp_true)/(sigma_cp + 1e-6)), axis=1)
        return chi2_amp + chi2_cp

    @tf.function
    def grad_log_likelihood_v2(self, Y_pred, X, alpha_amp=1., alpha_vis=1., alpha_bis=None, alpha_cp=1.):
        """
        :param Y_pred: reconstructed image
        :param X: interferometric data from measurements (complex vector from forward method)
        """
        # sigma_amp, sigma_bis, sigma_cp = self.get_std(X)
        grad = alpha_amp * chisqgrad_amp(Y_pred, self.A, tf.math.abs(X[..., :self.p]), 1/self.SNR, self.pixels)
        # grad = grad + alpha_vis * chisqgrad_vis(Y_pred, self.A, X[..., :self.p], 1/self.SNR, self.pixels)
        # if alpha_bis is not None:
        #     grad = grad + alpha_bis * chisqgrad_bs(Y_pred, self.A1, self.A2, self.A3, X[..., self.p:], sigma_bis, self.pixels)
        grad = grad + alpha_cp * chisqgrad_cphase(Y_pred, self.A1, self.A2, self.A3, tf.math.angle(X[..., self.p:]), self.phase_std, self.pixels)
        return grad

    @tf.function
    def grad_log_likelihood_v3(self, Y_pred, X, alpha_amp=1., alpha_vis=None, alpha_bis=None, alpha_cp=1.):
        """
        :param Y_pred: reconstructed image
        :param X: interferometric data from measurements (complex vector from forward method)
        """
        # sigma_amp, sigma_bis, sigma_cp = self.get_std(X)
        grad_amp = chisqgrad_amp(Y_pred, self.A, tf.math.abs(X[..., :self.p]), 1/self.SNR, self.pixels)
        # grad = alpha_vis * chisqgrad_vis(Y_pred, self.A, X[..., :self.p], sigma_amp, self.pixels)
        # if alpha_bis is not None:
        #     grad = grad + alpha_bis * chisqgrad_bs(Y_pred, self.A1, self.A2, self.A3, X[..., self.p:], sigma_bis, self.pixels)
        grad_cp = chisqgrad_cphase(Y_pred, self.A1, self.A2, self.A3, tf.math.angle(X[..., self.p:]), self.phase_std, self.pixels)
        return tf.concat([grad_amp, grad_cp], axis=3)

    @tf.function
    def grad_log_likelihood_v1(self, Y_pred, X):
        with tf.GradientTape() as tape:
            tape.watch(Y_pred)
            likelihood = self.log_likelihood_v1(Y_pred, X)
        grad = tape.gradient(likelihood, Y_pred)
        return grad

    # def get_std(self, X):
    #     sigma_vis = tf.math.abs(X[..., :self.p]) / self.SNR  # note that SNR should be large (say >~ 10 for gaussian approximation to hold)
    #     V1 = tf.einsum("ij, ...j -> ...i", self.V1_projector, X[..., :self.p])
    #     V2 = tf.einsum("ij, ...j -> ...i", self.V2_projector, X[..., :self.p])
    #     V3 = tf.einsum("ij, ...j -> ...i", self.V3_projector, X[..., :self.p])
    #     B_amp = tf.cast(tf.math.abs(V1 * tf.math.conj(V2) * V3), dtype)  # same hack from bispectrum
    #     sigma_cp = tf.cast(tf.math.sqrt(3 / self.SNR**2), dtype)
    #     sigma_bis = B_amp * sigma_cp
    #     return sigma_vis, sigma_bis, sigma_cp

    def simulate_noisy_data(self, images):
        """
        Noise is added in the form of a amplitude gain sampled from a normal distribution centered at 1.
        with
        """
        batch = images.shape[0]
        X = self.forward(images)
        # sigma_vis, sigma_bis, _ = self.get_std(X)
        # noise is picked from a complex normal distribution
        # vis_noise_real = tf.random.normal(shape=[batch, self.p], stddev=sigma_vis / 2, dtype=dtype)
        # vis_noise_imag = tf.random.normal(shape=[batch, self.p], stddev=sigma_vis / 2, dtype=dtype)
        # bis_noise_real = tf.random.normal(shape=[batch, self.q], stddev=sigma_bis / 2, dtype=dtype)
        # bis_noise_imag = tf.random.normal(shape=[batch, self.q], stddev=sigma_bis / 2, dtype=dtype)
        # vis_noise = tf.complex(vis_noise_real, vis_noise_imag)
        # bis_noise = tf.complex(bis_noise_real, bis_noise_imag)
        # noise = tf.concat([vis_noise, bis_noise], axis=1)
        gain = self._aperture_gain(batch)
        phase_error = self._visibility_phase_noise(batch)
        # add noise in polar form
        amp = tf.cast(gain * tf.math.abs(X[..., :self.p]), mycomplex)
        phase = 1j * tf.cast(tf.math.angle(X[..., :self.p]) + phase_error, mycomplex)
        noisy_vis = amp * tf.math.exp(phase)
        noisy_bis = self.bispectrum(noisy_vis)
        out = tf.concat([noisy_vis, noisy_bis], axis=1)
        return out

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