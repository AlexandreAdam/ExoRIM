from scipy.special import factorial
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # TODO consider using mixed precision with float16
mycomplex = tf.complex64
initializer = tf.random_normal_initializer(stddev=0.1)
DEGREE = tf.constant(3.14159265358979323 / 180., dtype)
INTENSITY_SCALER = tf.constant(1e6, dtype)
TWOPI = tf.constant(2*np.pi, dtype)


default_hyperparameters = {
        "steps": 12,
        "pixels": 32,
        "channels": 1,
        "state_size": 16,
        "state_depth": 32,
        "learning rate": {
            "initial_learning_rate": 1e-3,
            "decay_steps": 10000,
            "decay_rate": 0.90,
        },
        "Regularizer Amplitude": {
            "kernel": 0.01,
            "bias": 0.01
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
