from tensorflow.python.keras.layers.merge import concatenate
from astropy.cosmology import Planck15 as cosmo
from scipy.special import factorial
import tensorflow as tf
import ExoRIM.kpi as kpi
import numpy as np
import os

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
                "filters": 16,
                "strides": [2, 2]
            }}
        ],
        "Convolution Block": [
            {"Conv_1": {
                "kernel_size": [3, 3],
                "filters": 16,
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
                "filters": 16,
                "strides": [1, 1]
            }},
            {"TConv_2": {
                "kernel_size": [3, 3],
                "filters": 16,
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
