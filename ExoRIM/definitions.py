from tensorflow.python.keras.layers.merge import concatenate
from astropy.cosmology import Planck15 as cosmo
import tensorflow as tf
import ExoRIM.pysco.kpi as kpi
import numpy as np
import os

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # faster, otherise tf.float64
kernel_size = 3 # 3 or 1 for small input
initializer = tf.initializers.RandomNormal(stddev=0.06)  # random_normal_initializer(stddev=0.06)
kernal_reg_amp = 0.0
bias_reg_amp = 0.0
############### edit this line #################
basedir = os.path.abspath("/home/aadam/Desktop/Projects/ExoRIM")
################################################
datadir = os.path.join(basedir, "data")


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

