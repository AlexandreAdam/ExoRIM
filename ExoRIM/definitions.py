from scipy.special import factorial
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # faster, otherwise tf.float64
mycomplex = tf.complex64
initializer = tf.random_normal_initializer(stddev=0.1)
DEGREE = 3.14159265358979323 / 180.
INTENSITY_SCALER = tf.constant(1e6, dtype)
TWOPI = tf.constant(2*np.pi, dtype)

default_hyperparameters = {
        "steps": 12,
        "pixels": 32,
        "channels": 1,
        "state_size": 16,
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
                "filters": 16,
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


def triangle_pulse_f(omega, pdim):
    out = np.ones_like(omega)
    mask = np.where(omega != 0)[0]
    out[mask] *= (4.0/(pdim**2 * omega**2)) * (np.sin((pdim * omega)/2.0))**2
    return out


def rectangular_pulse_f(omega, pdim):
    out = np.ones_like(omega)
    mask = np.where(omega != 0)[0]
    out[mask] *= (2.0/(pdim * omega)) * (np.sin((pdim * omega)/2.0))
    return out


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


def log_scaling(X, base=10.):
    """
    This function separate positive and negative values of the tensor (axis=1 and 2) and computes their logarithm separately.
    """
    positive = X * tf.cast(X > 0, dtype) + 1e-5
    negative = - X * tf.cast(X < 0, dtype) + 1e-5
    positve = tf.clip_by_value(tf.math.log(positive) / tf.math.log(base), 0, 30)
    negative = - tf.clip_by_value(tf.math.log(negative) / tf.math.log(base), 0, 30)
    return positve + negative


def gradient_summary_log_scale(grad, base=10.):
    # assumes gradient have already been rescaled to the range [-1, 1], such that logs are always < 0
    positive = grad * tf.cast(grad > 0, dtype) + 1e-12
    negative = - grad * tf.cast(grad < 0, dtype) + 1e-12
    positve = - tf.math.log(positive) / tf.math.log(base)
    negative = tf.math.log(negative) / tf.math.log(base)
    return softmax_scaler(positve + negative, 0, 1)

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


def bispectrum(image, A1, A2, A3):
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B = V1 * tf.math.conj(V2) * V3
    return B


def chisq_vis(image, A, vis, sigma):
    """
    Chi squared of the complex visibilities.
        image: batch of square matrix (3-tensor)
        A: 2-tensor of phasor for the Fourier Transform (NDFTM operator)
        vis: data product of visibilties (dtype must be mycomplex)
        sigma: (0,1,2)-tensor. Inverse of the covariance matrix of the visibilities.
    """
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    diff = samples - vis
    if len(sigma.shape) < 2:
        chisq = 0.5 * tf.reduce_mean((tf.math.abs(diff) / sigma) ** 2, axis=1)
    else:
        sigma = tf.cast(sigma, mycomplex)
        chisq = 0.5 * tf.einsum("...j, ...j -> ...", tf.einsum("...i, ij,  -> ...j", sigma, diff), tf.math.conj(diff))
        chisq /= vis.shape[1]
    return chisq


def chisqgrad_vis(image, A, vis, sigma):
    """
    The gradient of the Chi squared of the complex visibilities relative to the image pixels. This is the analytical
    version, which computes much faster than the AutoGrad version.
    Also, this version only support diagonal covariance matrix, given in the form of a (0,1)-tensor. For a 2-tensor
    covariance matrix, AutoGrad should be used since this analytical version is no longer valid.
    """
    sig = tf.cast(sigma, mycomplex)  # prevent dividing by zero
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    wdiff = (vis - samples)/(sig**2)
    out = -tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(A), wdiff))
    out = tf.reshape(out, image.shape)
    return out / vis.shape[1]


def chisqgrad_vis_auto(image, A, vis, sigma):
    """
    The gradient of the Chi squared of the complex visibilities relative to the image pixels. This is the
    AutoGrad version.
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chisq_vis(image, A, vis, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_vis_phases(image, A, vphases, sigma):
    pass

def chisq_vis_phases_v2(image, A, vphases, sigma):
    im = cast_to_complex_flatten(image)
    vphases_samples = tf.math.angle(tf.einsum("ij, ...j -> ...i", A, im)) % TWOPI
    diff = vphases - vphases_samples
    if len(sigma.shape) < 2:
        chisq = tf.reduce_mean(tf.math.square(diff / sigma), axis=1)
    else:
        chisq = tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
        chisq /= vphases.shape[1]
    return chisq


def chisq_amp(image, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    return tf.math.reduce_mean(((amp - amp_samples)/sig)**2, axis=1)


def chisqgrad_amp(image, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""
    im = cast_to_complex_flatten(image)
    V_samples = tf.einsum("ij, ...j -> ...i", A, im)
    amp_samples = tf.math.abs(V_samples)
    product = (amp - amp_samples) / (sigma)**2 / amp_samples
    product = tf.cast(product, mycomplex)
    out = - 2.0 * tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(A), V_samples * product))
    out = tf.reshape(out, shape=image.shape)
    return out / amp.shape[1]


def chisq_bs(image, A1, A2, A3, B, sigma):
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B_sample = V1 * tf.math.conj(V2) * V3
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(B - B_sample)/sig), axis=1)
    return chisq


def chisqgrad_bs(image, A1, A2, A3, B, sigma):
    """The gradient of the bispectrum chi-squared"""
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    t_einsum = "ji, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B_samples = V1 * tf.math.conj(V2) * V3
    wdiff = tf.math.conj(B - B_samples)/(sig)**2
    out = tf.einsum(t_einsum, A1, wdiff * V2 * V3)
    out += tf.einsum(t_einsum, A2, wdiff * V1 * V3)
    out += tf.einsum(t_einsum, A3, wdiff * V1 * V2)
    out = -tf.math.real(out) / B.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out


def chisqgrad_bs_auto(image, A1, A2, A3, B, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chisq_bs(image, A1, A2, A3, B, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_cphase(image, A1, A2, A3, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    clphase_samples = tf.math.angle(V1 * tf.math.conj(V2) * V3)
    chisq = tf.reduce_mean(((1 - tf.math.cos(clphase - clphase_samples)) / sig)**2, axis=1)
    return chisq


def chisq_cphase_v2(image, A, CPO, clphase, sigma):
    """
    The A operator is the discrete fourier transform matrix.
    CPO is the closure phase operator.
    Closure phases chi squared with optional non-diagonal covariance matrix inverse (sigma). The chis squared depends
    directly on the closure phases, and not on the bispectrum phasors. Thus, the phase wrapping in the range [0, 2pi) is
    explicit in this derivation.
    Note that this version chooses to work with the linear operator instead of the Bispectrum.
    """
    im = cast_to_complex_flatten(image)
    phi = tf.math.angle(tf.einsum("ij, ...j -> ...i", A, im)) % TWOPI
    clphase_sample = tf.einsum("ij, ...j -> ...i", CPO, phi) % TWOPI
    diff = clphase - clphase_sample
    if len(sigma.shape) < 2:
        chisq = 0.5 * tf.reduce_mean((diff/sigma)**2, axis=1)
    else:
        chisq = 0.5 * tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
    return chisq


def chisqgrad_cphase(image, A1, A2, A3, clphase, sigma):
    """The gradient of the closure phase chi-squared for a sigma (0,1)-tensor"""
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B = V1 * tf.math.conj(V2) * V3
    clphase_samples = tf.math.angle(B)
    wdiff = tf.cast(tf.math.sin(clphase - clphase_samples) / sigma ** 2, mycomplex)
    out = tf.einsum("ji, ...j -> ...i", tf.math.conj(A1), wdiff / tf.math.conj(V1))
    out = out + tf.einsum("ji, ...j -> ...i", A2, wdiff / V2)
    out = out + tf.einsum("ji, ...j -> ...i", tf.math.conj(A3), wdiff / tf.math.conj(V3))
    out = -2. * tf.math.imag(out) / B.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out


def chisqgrad_cphase_auto(image, A1, A2, A3, clphase, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chisq_cphase(image, A1, A2, A3, clphase, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisqgrad_cphase_v2_auto(image, A, CPO, clphase, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chisq_cphase_v2(image, A, CPO, clphase, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


