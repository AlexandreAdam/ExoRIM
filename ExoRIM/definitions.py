from scipy.special import factorial
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # faster, otherise tf.float64
initializer = tf.initializers.GlorotNormal()  # random_normal_initializer(stddev=0.06)

default_hyperparameters = {
        "steps": 12,
        "pixels": 32,
        "channels": 1,
        "state_size": 8,
        "state_depth": 32,
        "Regularizer Amplitude": {
            "kernel": 0.01,
            "bias": 0.01
        },
        "Physical Model": {
            "Visibility Noise": 1e-4,
            "closure phase noise": 1e-5
        },
        "Downsampling Block": [
            {"Conv_Downsample": {
                "kernel_size": [3, 3],
                "filters": 1,
                "strides": [2, 2]
            }}
        ],
        "Convolution Block": [
            {"Conv_1": {
                "kernel_size": [3, 3],
                "filters": 8,
                "strides": [1, 1]
            }},
            {"Conv_2": {
                "kernel_size": [3, 3],
                "filters": 8,
                "strides": [1, 1]
            }}
        ],
        "Recurrent Block": {
            "GRU_1": {
                "kernel_size": [3, 3],
                "filters": 16
            },
            "Hidden_Conv_1": {
                "kernel_size": [3, 3],
                "filters": 16
            },
            "GRU_2": {
                "kernel_size": [3, 3],
                "filters": 16
            }
        },
        "Upsampling Block": [
            {"Conv_Fraction_Stride": {
                "kernel_size": [3, 3],
                "filters": 16,
                "strides": [2, 2]
            }}
        ],
        "Transposed Convolution Block": [
            {"TConv_1": {
                "kernel_size": [3, 3],
                "filters": 8,
                "strides": [1, 1]
            }},
            {"TConv_2": {
                "kernel_size": [3, 3],
                "filters": 1,
                "strides": [1, 1]
            }}
        ]
    }



def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus(-x - 5.0)


def xsquared(x):
    return (x/4)**2


def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))


def poisson(k, mu):
    return np.exp(-mu) * mu**k / factorial(k)


def k_truncated_poisson(k, mu):
    probabilities = poisson(k, mu)
    return probabilities / probabilities.sum()


def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return x * 4.8481368110953599e-09


# modified from F. Martinache Xara project
def one_sided_DFTM(uv, wavelength, pixels, plate_scale , inv=False, dprec=True):
    ''' ------------------------------------------------------------------
    Single-sided DFT matrix to be used with the "LDFT1" extraction method,
    DFT matrix computed for exact u (or v) coordinates.
    Based on a FF.dot(img) approach.
    parameters:
    ----------
    - uv : vector of baseline (u,v) coordinates where to compute the FT
    - wavelength: wavelength of light observed (in meters)
    - pixels: number of pixels of mage grid (on a side)
    - plate_scale: Plate scale of the telescope in mas/pixel (that is 206265/(1000 * f[mm] * pixel_density[pixel/mm]))
    Option:
    ------
    - inv    : Boolean (default=False) : True -> computes inverse DFT matrix
    - dprec  : double precision (default=True)
    For an image of size (SZ x SZ), the computation requires what can be a
    fairly large (N_UV x SZ^2) auxilliary matrix.
    -----------------------------------
    Example of use, for an image of size isz:

    >> FF = xara.core.compute_DFTM1(np.unique(kpi.UVC), m2pix, isz)
    >> FT = FF.dot(img.flatten())
    This last command returns a 1D vector FT of the img.
    ------------------------------------------------------------------ '''
    # e.g.
    # cwavel = 0.5e-6 # Wavelength [m]
    # ISZ = 128# Array size (number of pixel on a side)
    # pscale = 0.1 # plate scale [mas/pixel]
    m2pix = mas2rad(plate_scale) * pixels / wavelength
    i2pi = 1j * 2 * np.pi
    mydtype = np.complex64
    if dprec is True:
        mydtype = np.complex128
    xx, yy = np.meshgrid(np.arange(pixels) - pixels / 2, np.arange(pixels) - pixels / 2)
    uvc = uv * m2pix
    nuv = uvc.shape[0]

    if inv is True:
        WW = np.zeros((pixels ** 2, nuv), dtype=mydtype)
        for i in range(nuv):
            WW[:, i] = np.exp(i2pi * (uvc[i, 0] * xx.flatten() +
                                      uvc[i, 1] * yy.flatten()) / float(pixels))
    else:
        WW = np.zeros((nuv, pixels ** 2), dtype=mydtype)

        for i in range(nuv):
            WW[i] = np.exp(-i2pi * (uvc[i, 0] * xx.flatten() +
                                    uvc[i, 1] * yy.flatten()) / float(pixels))
    return WW


#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>
#  Alexandre Adam [2020-07-02]: modifications of chisq functions to work with tensors

##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def cast_to_complex_flatten(image):
    im = tf.dtypes.cast(image, tf.complex128)
    im = tf.keras.layers.Flatten(data_format="channel_last")(im)
    return im


def chisq_vis(image, A, vis, sigma):
    """Visibility chi-squared"""
    im = cast_to_complex_flatten(image)
    samples = tf.tensordot(A, im, axes=1)
    chisq = tf.reduce_mean(tf.math.abs((samples-vis)/sigma)**2)
    return chisq


def chisqgrad_vis(image, A, vis, sigma):
    """The gradient of the visibility chi-squared"""
    im = tf.dtypes.cast(image, tf.complex128)
    im = tf.keras.layers.Flatten(data_format="channel_last")(im)
    samples = tf.tensordot(A, im, axes=1)
    wdiff = (vis - samples)/(sigma**2)
    adjoint = tf.transpose(tf.math.conj(A))
    out = -tf.math.real(tf.tensordot(adjoint, wdiff, axes=1))
    return out


def chisq_amp(image, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""
    im = tf.dtypes.cast(image, tf.complex128)
    amp_samples = tf.math.abs(tf.tensordot(A, im, axes=1))
    return tf.math.reduce_mean(tf.math.abs((amp - amp_samples)/sigma)**2)


def chisqgrad_amp(image, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    im = tf.dtypes.cast(image, tf.complex128)
    V_samples = tf.tensordot(A, im, axes=1)
    amp_samples  = tf.math.abs(V_samples)
    product = ((amp - amp_samples) * amp_samples) / (sigma**2) / V_samples
    adjoint = tf.transpose(tf.math.conj(A))
    out = tf.reduce_mean(-2.0 * tf.math.real(tf.tensordot(adjoint, product, axes=1)))
    return out


def chisq_bs(image, A, B, bis, sigma):
    """

    :param image: Image tensor
    :param A: A matrix to project from image space to fourier space (one sided fourier transform)
    :param B: Bispectrum projection matrix to project from complex visibilities to bispectrum ()
    :param bis:
    :param sigma:
    :return:
    """
    bisamples = (np.dot(Amatrices[0], imvec) *
                 np.dot(Amatrices[1], imvec) *
                 np.dot(Amatrices[2], imvec))
    chisq = np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq


def chisqgrad_bs(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    bisamples = (np.dot(Amatrices[0], imvec) *
                 np.dot(Amatrices[1], imvec) *
                 np.dot(Amatrices[2], imvec))

    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    pt2 = wdiff * np.dot(Amatrices[0], imvec) * np.dot(Amatrices[2], imvec)
    pt3 = wdiff * np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec)
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]))

    out = -np.real(out) / len(bis)
    return out

DATATERMS = ['vis', 'amp', 'bs', 'cphase', 'cphase_diag',
             'camp', 'logcamp', 'logcamp_diag', 'logamp']
REGULARIZERS = ['gs', 'tv', 'tv2', 'l1w', 'lA', 'patch', 'simple', 'compact', 'compact2', 'rgauss']

def chisq_cphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq


def chisqgrad_cphase(imvec, Amatrices, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1 = pref/i1
    pt2 = pref/i2
    pt3 = pref/i3
    out = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])
    out = (-2.0/len(clphase)) * np.imag(out)
    return out


def chisq_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared"""
    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    clphase_diag_samples = []
    for iA, A3 in enumerate(A3_diag):
        clphase_samples = np.angle(np.dot(A3[0], imvec) *
                                   np.dot(A3[1], imvec) *
                                   np.dot(A3[2], imvec))
        clphase_diag_samples.append(np.dot(tform_mats[iA], clphase_samples))
    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    chisq *= (2.0/len(clphase_diag))
    return chisq


def chisqgrad_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """The gradient of the diagonalized closure phase chi-squared"""
    clphase_diag = clphase_diag * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A3 in enumerate(A3_diag):

        i1 = np.dot(A3[0], imvec)
        i2 = np.dot(A3[1], imvec)
        i3 = np.dot(A3[2], imvec)
        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples = np.dot(tform_mats[iA], clphase_samples)

        clphase_diag_measured = clphase_diag[iA]
        clphase_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i1)), A3[0])
        term2 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i2)), A3[1])
        term3 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i3)), A3[2])
        deriv += -2.0*np.imag(term1 + term2 + term3)

    deriv *= 1.0/np.float(len(np.concatenate(clphase_diag)))

    return deriv


def chisq_camp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq


def chisqgrad_camp(imvec, Amatrices, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 = pp/i1
    pt2 = pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))
    out *= (-2.0/len(clamp)) * np.real(out)
    return out


def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(np.dot(Amatrices[0], imvec))
    a2 = np.abs(np.dot(Amatrices[1], imvec))
    a3 = np.abs(np.dot(Amatrices[2], imvec))
    a4 = np.abs(np.dot(Amatrices[3], imvec))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return chisq


def chisqgrad_logcamp(imvec, Amatrices, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    log_clamp_samples = (np.log(np.abs(i1)) +
                         np.log(np.abs(i2)) -
                         np.log(np.abs(i3)) -
                         np.log(np.abs(i4)))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))
    out = (-2.0/len(log_clamp)) * np.real(out)
    return out


def chisq_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared"""

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    log_clamp_diag_samples = []
    for iA, A4 in enumerate(A4_diag):

        a1 = np.abs(np.dot(A4[0], imvec))
        a2 = np.abs(np.dot(A4[1], imvec))
        a3 = np.abs(np.dot(A4[2], imvec))
        a4 = np.abs(np.dot(A4[3], imvec))

        log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        log_clamp_diag_samples.append(np.dot(tform_mats[iA], log_clamp_samples))

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2)
    chisq /= (len(log_clamp_diag))

    return chisq


def chisqgrad_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared"""

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A4 in enumerate(A4_diag):

        i1 = np.dot(A4[0], imvec)
        i2 = np.dot(A4[1], imvec)
        i3 = np.dot(A4[2], imvec)
        i4 = np.dot(A4[3], imvec)
        log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - \
            np.log(np.abs(i3)) - np.log(np.abs(i4))
        log_clamp_diag_samples = np.dot(tform_mats[iA], log_clamp_samples)

        log_clamp_diag_measured = log_clamp_diag[iA]
        log_clamp_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i1)), A4[0])
        term2 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i2)), A4[1])
        term3 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i3)), A4[2])
        term4 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i4)), A4[3])
        deriv += -2.0*np.real(term1 + term2 - term3 - term4)

    deriv *= 1.0/np.float(len(np.concatenate(log_clamp_diag)))

    return deriv


def chisq_logamp(imvec, A, amp, sigma):
    """Log Visibility Amplitudes (normalized) chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    amp_samples = np.abs(np.dot(A, imvec))
    chisq = np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)
    return chisq

def chisqgrad_logamp(imvec, A, amp, sigma):
    """The gradient of the Log amplitude chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = ((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

