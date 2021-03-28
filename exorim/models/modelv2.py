import tensorflow as tf
from exorim.definitions import DTYPE
from exorim.models.layers import ConvGRU, ConvDownsampleBlock, TransposedConvBlock, ConvUpsampleBlock, ConvBlock
from exorim.models.utils import global_step, summary_histograms


class Model(tf.keras.models.Model):
    def __init__(self, activation="relu", kernel_reg_amp=0, bias_reg_amp=0, dtype=DTYPE, **kwargs):
        super(Model, self).__init__(dtype=dtype, name="Modelv2", **kwargs)
        self._timestep_mod = -1  # silent instance attribute to be modified if needed in the RIM fit method
        #TODO export hparams to json file for model versioning
        self.downsample1 = ConvDownsampleBlock(
                kernel_size=7,
                filters=16,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.downsample2 = ConvDownsampleBlock(
                kernel_size=5,
                filters=32,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
        )
        self.conv1 = ConvBlock(
                kernel_size=5,
                filters=16,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.conv2 = ConvBlock(
                kernel_size=5,
                filters=32,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.tconv1 = TransposedConvBlock(
                kernel_size=5,
                filters=64,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.tconv2 = TransposedConvBlock(
                kernel_size=5,
                filters=1,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.upsample1 = ConvUpsampleBlock(
                kernel_size=5,
                filters=64,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.upsample2 = ConvUpsampleBlock(
                kernel_size=5,
                filters=16,
                kernel_reg_amp=kernel_reg_amp,
                bias_reg_amp=bias_reg_amp,
                activation=activation
            )
        self.gru1 = ConvGRU(filters=128, kernel_size=3)
        self.gru2 = ConvGRU(filters=128, kernel_size=3)
        self.hidden_conv = tf.keras.layers.Conv2D(
            kernel_size=5,
            filters=128,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            activation=activation
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

