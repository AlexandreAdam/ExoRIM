import tensorflow as tf


def summary_histograms(layer, activation):
    tf.summary.histogram(layer.name + "_activation", data=activation, step=tf.summary.experimental.get_step())
    for weights in layer.trainable_weights:
        tf.summary.histogram(weights.name, data=weights, step=tf.summary.experimental.get_step())


def global_step():
    step = tf.summary.experimental.get_step()
    if step is not None:
        return step
    else:
        return -1
