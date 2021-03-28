import tensorflow as tf
from exorim.definitions import DTYPE


class ConvDownsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, kernel_reg_amp, bias_reg_amp, **kwargs):
        super(ConvDownsampleBlock, self).__init__(DTYPE, **kwargs)
        self.non_linearity = activation
        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization(scale=False)

        self.conv_rescale1 = tf.keras.layers.Conv2D(
            filters=filters//2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.RandomUniform()
        )
        self.conv_rescale2 = tf.keras.layers.Conv2D(
            filters=filters//2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.RandomUniform()
        )

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
        )

    def call(self, X):
        X = self.batch_norm1(X)
        X = self.non_linearity(X)
        X = self.conv_rescale1(X)
        X = self.batch_norm2(X)
        X = self.non_linearity(X)
        X = self.conv(X)
        X = self.batch_norm3(X)
        X = self.non_linearity(X)
        X = self.conv_rescale2(X)
        return X
