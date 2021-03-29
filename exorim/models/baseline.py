import tensorflow as tf
from exorim.definitions import DTYPE
from exorim.models.layers import ConvGRU
from exorim.models.utils import global_step, summary_histograms


class BaselineModel(tf.keras.models.Model):
    def __init__(self, dtype=DTYPE, **kwargs):
        super(BaselineModel, self).__init__(dtype=dtype, name="BaselineModel", **kwargs)
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        self.downsample1 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                strides=2,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.downsample2 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                strides=2,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.conv1 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                strides=1,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.conv2 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                strides=1,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.tconv1 = tf.keras.layers.Conv2DTranspose(
                kernel_size=3,
                filters=16,
                strides=1,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.tconv2 = tf.keras.layers.Conv2DTranspose(
                kernel_size=3,
                filters=1,
                strides=2,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.upsample1 = tf.keras.layers.Conv2DTranspose(
                kernel_size=3,
                filters=16,
                strides=2,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.upsample2 = tf.keras.layers.Conv2DTranspose(
                kernel_size=3,
                filters=16,
                strides=2,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.gru1 = ConvGRU(filters=256, kernel_size=3)
        self.gru2 = ConvGRU(filters=256, kernel_size=3)
        self.hidden_conv = tf.keras.layers.Conv2D(
            kernel_size=3,
            filters=16,
            strides=1,
            activation=tf.keras.layers.LeakyReLU(),
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, X, ht):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        X = self.downsample1(X)
        X = self.conv1(X)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.conv1, X)
        X = self.downsample2(X)
        X = self.conv2(X)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.conv2, X)

        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(X, ht_1)  # to be recombined in new state
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru1, ht_1)
        ht_1_features = self.hidden_conv(ht_1)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.hidden_conv, ht_1_features)
        ht_2 = self.gru2(ht_1_features, ht_2)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru2, ht_2)
        # ===========================

        X = self.upsample1(ht_2)
        X = self.tconv1(X)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.tconv1, X)
        X = self.upsample2(X)
        X = self.tconv2(X)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.tconv2, X)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return X, new_state
