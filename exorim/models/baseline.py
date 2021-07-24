import tensorflow as tf
from exorim.definitions import DTYPE
from exorim.models.layers import ConvGRU


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
        self.tconv1 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=16,
                strides=1,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        self.tconv2 = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=1,
                strides=1,
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
        self.state_depth = 32
        self.downsampling_factor = 4
        self.gru1 = ConvGRU(filters=16, kernel_size=3)
        self.gru2 = ConvGRU(filters=16, kernel_size=3)
        self.hidden_conv = tf.keras.layers.Conv2D(
            kernel_size=3,
            filters=32,
            strides=1,
            activation=tf.keras.layers.LeakyReLU(),
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, xt, ht, grad):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        delta_xt = tf.concat([tf.identity(xt), grad], axis=3)
        delta_xt = self.downsample1(delta_xt)
        delta_xt = self.conv1(delta_xt)
        delta_xt = self.downsample2(delta_xt)
        delta_xt = self.conv2(delta_xt)

        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(delta_xt, ht_1)  # to be recombined in new state
        ht_1_features = self.hidden_conv(ht_1)
        ht_2 = self.gru2(ht_1_features, ht_2)
        # ===========================

        delta_xt = self.upsample1(ht_2)
        delta_xt = self.tconv1(delta_xt)
        delta_xt = self.upsample2(delta_xt)
        delta_xt = self.tconv2(delta_xt)

        new_state = tf.concat([ht_1, ht_2], axis=3)
        xt_1 = xt + delta_xt
        return xt_1, new_state

    def init_hidden_states(self, input_pixels, batch_size, constant=0.):
        state_size = input_pixels // self.downsampling_factor
        return constant * tf.ones(shape=(batch_size, state_size, state_size, self.state_depth), dtype=self._dtype)
