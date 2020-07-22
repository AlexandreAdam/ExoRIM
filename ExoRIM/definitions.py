from scipy.special import factorial
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # faster, otherwise tf.float64
mycomplex = tf.complex64
initializer = tf.random_normal_initializer(stddev=0.1)
DEGREE = 3.14159265358979323 / 180.
INTENSITY_SCALER = tf.constant(1e6, dtype)

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


def rad2mas(x):
    '''  convert radians to mas'''
    return(x / 4.8481368110953599e-09) # = x / (np.pi/(180*3600*1000))


def pixel_grid(pixels, symmetric=True):
    x = np.arange(pixels, dtype=np.float32) - pixels // 2
    xx, yy = np.meshgrid(x, x)
    if pixels % 2 == 0 and symmetric:
        # make pixel grid symmetric
        xx += 0.5
        yy += 0.5
    return xx, yy


def centroid(image, threshold=0, binarize=False):
    ''' ------------------------------------------------------
    Determines the center of gravity of an array
    Parameters:
    ----------
    - image: the array
    - threshold: value above which pixels are taken into account
    - binarize: binarizes the image before centroid (boolean)
    Remarks:
    -------
    The binarize option can be useful for apertures, expected
    to be uniformly lit.
    ------------------------------------------------------ '''

    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1]  # size of "image"
    bkg_cnt = np.median(image)

    temp = np.zeros((sy, sx))

    if binarize is True:
        temp[signal] = 1.0
    else:
        temp[signal] = image[signal]

    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)
    profx -= np.min(profx)
    profy -= np.min(profy)

    x0 = (profx * np.arange(sx)).sum() / profx.sum()
    y0 = (profy * np.arange(sy)).sum() / profy.sum()

    return (x0, y0)


def logsumexp_scaler(X, minimum, maximum, bkg=0):
    """
    This function implement a scaler with a smooth maximum functions for gradient operations.
    Our smooth maximum operator is a poor approximation of the maximum function in this range, and so we rescale
    by multiplying by a large number (INTENSITY_SCALER)

    Finally, this function assumes the minimum of X to be 0. It can be corrected using background variable bkg
    """
    # X should be an image of shape (batch, pix, pix, channels)
    x_max = tf.reduce_logsumexp(INTENSITY_SCALER * X, axis=[1, 2, 3])
    x_max = tf.reshape(x_max, x_max.shape + [1, 1, 1]) # for broadcast sum
    return minimum + (INTENSITY_SCALER * X - bkg) * (maximum - minimum) / (x_max - bkg)


def softmax_scaler(X, minimum, maximum):
    """
    Implement a more versatile scaler using softmax, which can be used to approximate both a maximum and a
    minimum smooth function by adjusting the alpha parameter.
    """
    dims = len(X.shape)
    if dims > 2:
        _X = tf.keras.layers.Flatten(data_format="channels_last")(X)
    else:
        _X = X
    # Note that this smooth maximum approximator tends to have better results for alpha -> infinity
    alpha = INTENSITY_SCALER
    # X dot softmax(alpha * X) broadcasted over batch dimension
    x_max = tf.einsum("...i, ...i -> ...", _X, tf.math.softmax(alpha * _X))
    x_max = tf.reshape(x_max, x_max.shape + [1]*(dims - 1))  # hack to broadcast sum
    x_min = tf.einsum("...i, ...i -> ...", _X, tf.math.softmax(-alpha * _X))
    x_min = tf.reshape(x_min, x_min.shape + [1]*(dims - 1))
    return minimum + (X - x_min) * (maximum - minimum) / (x_max - x_min + 1e-8)


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
    im = tf.dtypes.cast(image, mycomplex)
    im = tf.keras.layers.Flatten(data_format="channels_last")(im)
    return im


def chisq_vis(image, A, vis, sigma):
    """Visibility chi-squared"""
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    chisq = 0.5 * tf.reduce_mean(tf.math.abs((samples - vis) / sig)**2, axis=1)
    return chisq


def chisqgrad_vis(image, A, vis, sigma, pix, floor=1e-6):
    """The gradient of the visibility chi-squared"""
    sig = tf.cast(sigma + floor, mycomplex)  # prevent dividing by zero
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    wdiff = (vis - samples)/(sig**2)
    out = -tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(A), wdiff))
    out = tf.reshape(out, [-1, pix, pix, 1])
    return out


def chisq_amp(image, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    return tf.math.reduce_mean(tf.math.abs((amp - amp_samples)/sig)**2, axis=1)


def chisqgrad_amp(image, A, amp, sigma, pix, floor=1e-6):
    """The gradient of the amplitude chi-squared"""
    im = cast_to_complex_flatten(image)
    V_samples = tf.einsum("ij, ...j -> ...i", A, im)
    amp_samples = tf.math.abs(V_samples)
    product = ((amp - amp_samples)) / ((sigma + floor)**2) / amp_samples
    product = tf.cast(product, mycomplex)
    adjoint = tf.transpose(tf.math.conj(A))
    out = -2.0 * tf.math.real(tf.einsum("ij, ...j -> ...i", adjoint, V_samples * product))
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out / amp.shape[1]


def chisq_bs(image, A1, A2, A3, B, sigma):
    """

    :param image: Image tensor
    :param Amatrices:
    :param B: Bispectrum projection matrix to project from complex visibilities to bispectrum ()
    :param bis:
    :param sigma:
    :return:
    """
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B_sample = V1 * V2 * V3
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(B - B_sample)/sig), axis=1)
    return chisq


def chisqgrad_bs(image, A1, A2, A3, B, sigma, pix, floor=1e-6):
    """The gradient of the bispectrum chi-squared"""
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    t_einsum = "ji, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B_samples = V1 * V2 * V3
    wdiff = tf.math.conj(B - B_samples)/(sig + floor)**2
    out = tf.einsum(t_einsum, A1, wdiff * V2 * V3) + tf.einsum(t_einsum, A2, wdiff * V1 * V3) + tf.einsum(t_einsum, A3, wdiff * V1 * V2)
    out = -tf.math.real(out) / B.shape[1]
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out

# DATATERMS = ['vis', 'amp', 'bs', 'cphase', 'cphase_diag',
#              'camp', 'logcamp', 'logcamp_diag', 'logamp']
# REGULARIZERS = ['gs', 'tv', 'tv2', 'l1w', 'lA', 'patch', 'simple', 'compact', 'compact2', 'rgauss']
#


def chisq_cphase(image, A1, A2, A3, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    clphase_samples = tf.math.angle(V1 * V2 * V3)
    chisq = tf.reduce_mean((1.0 - tf.math.cos(clphase-clphase_samples) / sig**2), axis=1)
    return chisq


def chisqgrad_cphase(image, A1, A2, A3, clphase, sigma, pix):
    """The gradient of the closure phase chi-squared"""

    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    t_einsum = "ji, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B = V1 * tf.math.conj(V2) * V3
    clphase_samples = tf.math.angle(B)
    wdiff = tf.cast(tf.math.sin(clphase - clphase_samples)/sigma**2, mycomplex)
    out = tf.einsum(t_einsum, tf.math.conj(A1), wdiff / tf.math.conj(V1))
    out = out + tf.einsum(t_einsum, A2, wdiff / V2)
    out = out + tf.einsum(t_einsum, tf.math.conj(A3), wdiff / tf.math.conj(V3))
    out = -2. * tf.math.imag(out) / B.shape[1]
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out

#
# def chisq_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
#     """Diagonalized closure phases (normalized) chi-squared"""
#     clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
#     sigma = np.concatenate(sigma) * ehc.DEGREE
#
#     A3_diag = Amatrices[0]
#     tform_mats = Amatrices[1]
#
#     clphase_diag_samples = []
#     for iA, A3 in enumerate(A3_diag):
#         clphase_samples = np.angle(np.dot(A3[0], imvec) *
#                                    np.dot(A3[1], imvec) *
#                                    np.dot(A3[2], imvec))
#         clphase_diag_samples.append(np.dot(tform_mats[iA], clphase_samples))
#     clphase_diag_samples = np.concatenate(clphase_diag_samples)
#
#     chisq = np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
#     chisq *= (2.0/len(clphase_diag))
#     return chisq
#
#
# def chisqgrad_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
#     """The gradient of the diagonalized closure phase chi-squared"""
#     clphase_diag = clphase_diag * ehc.DEGREE
#     sigma = sigma * ehc.DEGREE
#
#     A3_diag = Amatrices[0]
#     tform_mats = Amatrices[1]
#
#     deriv = np.zeros_like(imvec)
#     for iA, A3 in enumerate(A3_diag):
#
#         i1 = np.dot(A3[0], imvec)
#         i2 = np.dot(A3[1], imvec)
#         i3 = np.dot(A3[2], imvec)
#         clphase_samples = np.angle(i1 * i2 * i3)
#         clphase_diag_samples = np.dot(tform_mats[iA], clphase_samples)
#
#         clphase_diag_measured = clphase_diag[iA]
#         clphase_diag_sigma = sigma[iA]
#
#         term1 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
#                                (clphase_diag_sigma**2.0)), (tform_mats[iA]/i1)), A3[0])
#         term2 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
#                                (clphase_diag_sigma**2.0)), (tform_mats[iA]/i2)), A3[1])
#         term3 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
#                                (clphase_diag_sigma**2.0)), (tform_mats[iA]/i3)), A3[2])
#         deriv += -2.0*np.imag(term1 + term2 + term3)
#
#     deriv *= 1.0/np.float(len(np.concatenate(clphase_diag)))
#
#     return deriv
#
#
# def chisq_camp(imvec, Amatrices, clamp, sigma):
#     """Closure Amplitudes (normalized) chi-squared"""
#
#     i1 = np.dot(Amatrices[0], imvec)
#     i2 = np.dot(Amatrices[1], imvec)
#     i3 = np.dot(Amatrices[2], imvec)
#     i4 = np.dot(Amatrices[3], imvec)
#     clamp_samples = np.abs((i1 * i2)/(i3 * i4))
#
#     chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
#     return chisq
#
#
# def chisqgrad_camp(imvec, Amatrices, clamp, sigma):
#     """The gradient of the closure amplitude chi-squared"""
#
#     i1 = np.dot(Amatrices[0], imvec)
#     i2 = np.dot(Amatrices[1], imvec)
#     i3 = np.dot(Amatrices[2], imvec)
#     i4 = np.dot(Amatrices[3], imvec)
#     clamp_samples = np.abs((i1 * i2)/(i3 * i4))
#
#     pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
#     pt1 = pp/i1
#     pt2 = pp/i2
#     pt3 = -pp/i3
#     pt4 = -pp/i4
#     out = (np.dot(pt1, Amatrices[0]) +
#            np.dot(pt2, Amatrices[1]) +
#            np.dot(pt3, Amatrices[2]) +
#            np.dot(pt4, Amatrices[3]))
#     out *= (-2.0/len(clamp)) * np.real(out)
#     return out
#
#
# def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
#     """Log Closure Amplitudes (normalized) chi-squared"""
#
#     a1 = np.abs(np.dot(Amatrices[0], imvec))
#     a2 = np.abs(np.dot(Amatrices[1], imvec))
#     a3 = np.abs(np.dot(Amatrices[2], imvec))
#     a4 = np.abs(np.dot(Amatrices[3], imvec))
#
#     samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
#     chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
#     return chisq
#
#
# def chisqgrad_logcamp(imvec, Amatrices, log_clamp, sigma):
#     """The gradient of the Log closure amplitude chi-squared"""
#
#     i1 = np.dot(Amatrices[0], imvec)
#     i2 = np.dot(Amatrices[1], imvec)
#     i3 = np.dot(Amatrices[2], imvec)
#     i4 = np.dot(Amatrices[3], imvec)
#     log_clamp_samples = (np.log(np.abs(i1)) +
#                          np.log(np.abs(i2)) -
#                          np.log(np.abs(i3)) -
#                          np.log(np.abs(i4)))
#
#     pp = (log_clamp - log_clamp_samples) / (sigma**2)
#     pt1 = pp / i1
#     pt2 = pp / i2
#     pt3 = -pp / i3
#     pt4 = -pp / i4
#     out = (np.dot(pt1, Amatrices[0]) +
#            np.dot(pt2, Amatrices[1]) +
#            np.dot(pt3, Amatrices[2]) +
#            np.dot(pt4, Amatrices[3]))
#     out = (-2.0/len(log_clamp)) * np.real(out)
#     return out
#
#
# def chisq_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
#     """Diagonalized log closure amplitudes (normalized) chi-squared"""
#
#     log_clamp_diag = np.concatenate(log_clamp_diag)
#     sigma = np.concatenate(sigma)
#
#     A4_diag = Amatrices[0]
#     tform_mats = Amatrices[1]
#
#     log_clamp_diag_samples = []
#     for iA, A4 in enumerate(A4_diag):
#
#         a1 = np.abs(np.dot(A4[0], imvec))
#         a2 = np.abs(np.dot(A4[1], imvec))
#         a3 = np.abs(np.dot(A4[2], imvec))
#         a4 = np.abs(np.dot(A4[3], imvec))
#
#         log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
#         log_clamp_diag_samples.append(np.dot(tform_mats[iA], log_clamp_samples))
#
#     log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)
#
#     chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2)
#     chisq /= (len(log_clamp_diag))
#
#     return chisq
#
#
# def chisqgrad_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
#     """The gradient of the diagonalized log closure amplitude chi-squared"""
#
#     A4_diag = Amatrices[0]
#     tform_mats = Amatrices[1]
#
#     deriv = np.zeros_like(imvec)
#     for iA, A4 in enumerate(A4_diag):
#
#         i1 = np.dot(A4[0], imvec)
#         i2 = np.dot(A4[1], imvec)
#         i3 = np.dot(A4[2], imvec)
#         i4 = np.dot(A4[3], imvec)
#         log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - \
#             np.log(np.abs(i3)) - np.log(np.abs(i4))
#         log_clamp_diag_samples = np.dot(tform_mats[iA], log_clamp_samples)
#
#         log_clamp_diag_measured = log_clamp_diag[iA]
#         log_clamp_diag_sigma = sigma[iA]
#
#         term1 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
#                                (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i1)), A4[0])
#         term2 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
#                                (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i2)), A4[1])
#         term3 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
#                                (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i3)), A4[2])
#         term4 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
#                                (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i4)), A4[3])
#         deriv += -2.0*np.real(term1 + term2 - term3 - term4)
#
#     deriv *= 1.0/np.float(len(np.concatenate(log_clamp_diag)))
#
#     return deriv
#
#
# def chisq_logamp(imvec, A, amp, sigma):
#     """Log Visibility Amplitudes (normalized) chi-squared"""
#
#     # to lowest order the variance on the logarithm of a quantity x is
#     # sigma^2_log(x) = sigma^2/x^2
#     logsigma = sigma / amp
#
#     amp_samples = np.abs(np.dot(A, imvec))
#     chisq = np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)
#     return chisq
#
# def chisqgrad_logamp(imvec, A, amp, sigma):
#     """The gradient of the Log amplitude chi-squared"""
#
#     # to lowest order the variance on the logarithm of a quantity x is
#     # sigma^2_log(x) = sigma^2/x^2
#     logsigma = sigma / amp
#
#     i1 = np.dot(A, imvec)
#     amp_samples = np.abs(i1)
#
#     pp = ((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / i1
#     out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
#     return out
#
