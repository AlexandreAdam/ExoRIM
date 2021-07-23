import tensorflow as tf
from exorim.definitions import DTYPE
from exorim.models.layers.conv_gru import ConvGRU
from exorim.models.utils import global_step, summary_histograms
from .utils import get_activation


class Modelv1(tf.keras.models.Model):
    def __init__(self,
                 name="modelv1",
                 kernel_size=3,
                 filters=16,
                 layers=1,
                 block_conv_layers=1,
                 kernel_size_gru=3,
                 state_depth=64,
                 hidden_layers=1,
                 kernel_regularizer_amp=0.,
                 bias_regularizer_amp=0.,
                 batch_norm=False,
                 dtype=DTYPE,
                 activation="leaky_relu",
                 **kwargs):
        super(Modelv1, self).__init__(dtype=dtype, name=name, **kwargs)
        self.state_depth = state_depth
        self.downsampling_factor = 2**layers
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        self.downsampling_block = []
        self.recurrent_block = []
        self.upsampling_block = []
        self.hidden_conv = []
        activation = get_activation(activation)
        for i in range(layers):
            self.downsampling_block.append(tf.keras.layers.Conv2D(
                strides=2,
                kernel_size=kernel_size,
                filters=filters,
                name=f"DownsampleConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.downsampling_block.append(tf.keras.layers.BatchNormalization(name=f"BatchNormDownsample{i+1}", axis=-1))
            for j in range(block_conv_layers):
                self.downsampling_block.append(tf.keras.layers.Conv2D(
                    strides=1,
                    kernel_size=kernel_size,
                    filters=filters,
                    name=f"Conv{i+1}_{j+1}",
                    activation=activation,
                    padding="same",
                    kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                    bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                    data_format="channels_last",
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                ))
                if batch_norm:
                    self.downsampling_block.append(
                        tf.keras.layers.BatchNormalization(name=f"BatchNormDownsampleConv{j + 1}", axis=-1))
        for i in range(layers):
            self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                strides=2,
                kernel_size=kernel_size,
                filters=filters,
                name=f"UpsampleConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.upsampling_block.append(tf.keras.layers.BatchNormalization(name=f"BatchNormUpsample{i+1}", axis=-1))
            for j in range(layers):
                self.upsampling_block.append(tf.keras.layers.Conv2D(
                    strides=1,
                    kernel_size=kernel_size,
                    filters=filters,
                    name=f"TConv{i+1}_{j+1}",
                    activation=activation,
                    padding="same",
                    kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                    bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                    data_format="channels_last",
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                ))
                if batch_norm and (j != block_conv_layers-1 or i != layers-1): # except last layer
                    self.upsampling_block.append(
                        tf.keras.layers.BatchNormalization(name=f"BatchNormUpsampleConv{j + 1}", axis=-1))
        self.gru1 = ConvGRU(filters=self.state_depth//2, kernel_size=kernel_size_gru)
        self.gru2 = ConvGRU(filters=self.state_depth//2, kernel_size=kernel_size_gru)
        for i in range(hidden_layers):
            self.hidden_conv.append(tf.keras.layers.Conv2D(
                filters=self.state_depth,
                kernel_size=kernel_size_gru,
                name=f"HiddenConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.hidden_conv.append(
                    tf.keras.layers.BatchNormalization(name=f"BatchNormHiddenConv{i + 1}", axis=-1))
        self.output_layer = tf.keras.layers.Conv2DTranspose(
            name="output_conv",
            kernel_size=1,
            filters=1,
            activation=tf.keras.layers.Activation("linear"),
            padding="same",
        )

    def call(self, xt, ht, grad):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        delta_xt = tf.concat([tf.identity(xt), grad], axis=3)
        for layer in self.downsampling_block:
            delta_xt = layer(delta_xt)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, delta_xt)

        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(delta_xt, ht_1)  # to be recombined in new state
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru1, ht_1)
        ht_1_features = tf.identity(ht_1)
        for layer in self.hidden_conv:
            ht_1_features = layer(ht_1_features)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, ht_1_features)
        ht_2 = self.gru2(ht_1_features, ht_2)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru2, ht_2)
        # ===========================

        delta_xt = tf.identity(ht_2)
        for layer in self.upsampling_block:
            delta_xt = layer(delta_xt)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, delta_xt)
        delta_xt = self.output_layer(delta_xt)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.output_layer, delta_xt)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        xt_1 = xt + delta_xt  # update image
        return xt_1, new_state

    def init_hidden_states(self, input_pixels, batch_size, constant=0.):
        state_size = input_pixels // self.downsampling_factor
        return constant * tf.ones(shape=(batch_size, state_size, state_size, self.state_depth), dtype=self._dtype)