import tensorflow as tf
from exorim.definitions import DTYPE, default_hyperparameters
from exorim.models.layers.conv_gru import ConvGRU
from exorim.models.utils import global_step, summary_histograms


class Model(tf.keras.models.Model):
    def __init__(self, hyperparameters=default_hyperparameters, dtype=DTYPE, **kwargs):
        try:
            name = hyperparameters["name"]
        except KeyError:
            name = "modelv1"
        super(Model, self).__init__(dtype=dtype, name=name, **kwargs)
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        self.downsampling_block = []
        self.convolution_block = []
        self.recurrent_block = []
        self.upsampling_block = []
        self.transposed_convolution_block = []
        self.batch_norm = []
        try:
            batch_norm_params = hyperparameters["Batch Norm"]
        except KeyError:
            batch_norm_params = {}
        for i in range(5):
            self.batch_norm.append(tf.keras.layers.BatchNormalization(axis=-1, **batch_norm_params)) # last axis is channel dimension -> to be normalised
        kernel_reg_amp = hyperparameters["Regularizer Amplitude"]["kernel"]
        bias_reg_amp = hyperparameters["Regularizer Amplitude"]["bias"]
        for layer in hyperparameters["Downsampling Block"]:
            name = list(layer.keys())[0]
            params = layer[name]
            self.downsampling_block.append(tf.keras.layers.Conv2D(
                # stride=(2, 2),  # (output side pixel)/
                **params,  # kernel size and filters
                name=name,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
        for layer in hyperparameters["Convolution Block"]:
            name = list(layer.keys())[0]
            params = layer[name]
            self.convolution_block.append(tf.keras.layers.Conv2D(
                # stride=(1, 1),
                **params,
                name=name,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
        for layer in hyperparameters["Transposed Convolution Block"]:
            name = list(layer.keys())[0]
            params = layer[name]
            self.transposed_convolution_block.append(tf.keras.layers.Conv2DTranspose(
                # stride=(1, 1),
                **params,
                name=name,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
        for layer in hyperparameters["Upsampling Block"]:
            name = list(layer.keys())[0]
            params = layer[name]
            self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                **params,
                name=name,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
        if hyperparameters["Upsampling Block"] == []:
            name = "Identity"
            self.upsampling_block.append(tf.identity)
        self.gru1 = ConvGRU(**hyperparameters["Recurrent Block"]["GRU_1"])
        self.gru2 = ConvGRU(**hyperparameters["Recurrent Block"]["GRU_2"])
        if "Hidden_Conv_1" in hyperparameters["Recurrent Block"].keys():
            self.hidden_conv = tf.keras.layers.Conv2DTranspose(
                **hyperparameters["Recurrent Block"]["Hidden_Conv_1"],
                name="Hidden_Conv_1",
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        else:
            self.hidden_conv = None


    def call(self, X, ht):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        X = self.batch_norm[0](X)
        for layer in self.downsampling_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)
        X = self.batch_norm[1](X)
        for layer in self.convolution_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)
        X = self.batch_norm[2](X)
        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(X, ht_1)  # to be recombined in new state
        summary_histograms(self.gru1, ht_1)
        if self.hidden_conv is not None:
            ht_1_features = self.hidden_conv(ht_1)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(self.hidden_conv, ht_1_features)
        else:
            ht_1_features = ht_1
        ht_2 = self.gru2(ht_1_features, ht_2)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru2, ht_2)
        # ===========================
        delta_xt = self.batch_norm[3](ht_2)
        # delta_xt = self.upsample2d(delta_xt)
        for layer in self.upsampling_block:
            delta_xt = layer(delta_xt)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, delta_xt)
        delta_xt = self.batch_norm[4](delta_xt)
        for layer in self.transposed_convolution_block:
            delta_xt = layer(delta_xt)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, delta_xt)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return delta_xt, new_state