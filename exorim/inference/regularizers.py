import tensorflow as tf
from exorim.definitions import DTYPE

LOG_FLOOR = tf.constant(1e-16, DTYPE)

def entropy(image, prior):
    # prior should have dimension [1, pixel, pixel, 1]
    # safe log for numerical stability
    log = tf.math.log(image + LOG_FLOOR) - tf.math.log(prior + LOG_FLOOR)
    return tf.reduce_mean(image * log - image + prior)
