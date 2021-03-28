import tensorflow as tf
from exorim.definitions import DTYPE


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, kernel_reg_amp, bias_reg_amp, alpha=0.1, **kwargs):
        super(ResidualBlock, self).__init__(DTYPE, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha=alpha),
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
        )

    def call(self, X):
        features = self.conv1(X)
        features = self.conv2(features)
        return tf.add(X, features)
