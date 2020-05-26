from tensorflow.python.keras.layers.merge import concatenate
from astropy.cosmology import Planck15 as cosmo
from scipy.special import factorial
import tensorflow as tf
import ExoRIM.pysco.kpi as kpi
import numpy as np
import os

tf.keras.backend.set_floatx('float32')
dtype = tf.float32  # faster, otherise tf.float64
kernel_size = 3 # 3 or 1 for small input
initializer = tf.initializers.GlorotNormal()  # random_normal_initializer(stddev=0.06)
kernal_reg_amp = 0.01
bias_reg_amp = 0.01
basedir = os.getcwd() #os.path.abspath("/home/aadam/scratch/ExoRIM")
datadir = os.path.join(basedir, "data")
if not os.path.isdir(datadir):
    os.mkdir(datadir)
lossdir = os.path.join(datadir, "loss")
if not os.path.isdir(lossdir):
    os.mkdir(lossdir)
modeldir = os.path.join(basedir, "models")
if not os.path.isdir(modeldir):
    os.mkdir(modeldir)
image_dir = os.path.join(datadir, "generated_images")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

######## For the data generator ###############


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
    return np.exp(-mu) * mu ** k / factorial(k)

def k_truncated_poisson(k, mu):
    probabilities = poisson(k, mu)
    return probabilities / probabilities.sum()

