import tensorflow as tf
from ExoRIM.definitions import initializer, default_hyperparameters, dtype
from ExoRIM.utilities import nullwriter


class ConvGRU(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ConvGRU, self).__init__()
        self.update_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='sigmoid',
            padding='same',
            kernel_initializer=initializer
        )
        self.reset_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='sigmoid',
            padding='same',
            kernel_initializer=initializer
        )
        self.candidate_activation_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='tanh',
            padding='same',
            kernel_initializer=initializer
        )


    def call(self, features, ht):
        """
        Compute the new state tensor h_{t+1}.
        """
        stacked_input = tf.concat([features, ht], axis=3)
        z = self.update_gate(stacked_input)  # Update gate vector
        r = self.reset_gate(stacked_input)  # Reset gate vector
        r_state = tf.multiply(r, ht)
        stacked_r_state = tf.concat([features, r_state], axis=3)
        tilde_h = self.candidate_activation_gate(stacked_r_state)
        new_state = tf.multiply(z, ht) + tf.multiply(1 - z, tilde_h)
        return new_state  # h_{t+1}


class Model(tf.keras.Model):
    def __init__(self, hyperparameters=default_hyperparameters, dtype=dtype):
        super(Model, self).__init__(dtype=dtype)
        self.downsampling_block = []
        self.convolution_block = []
        self.recurrent_block = []
        self.upsampling_block = []
        self.transposed_convolution_block = []
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
                kernel_initializer=initializer
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
                kernel_initializer=initializer
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
                kernel_initializer=initializer
            ))
        for layer in hyperparameters["Upsampling Block"]:
            name = list(layer.keys())[0]
            params = layer[name]
            self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                # stride=(2, 2),  # stride of 1/4, pixel*4
                **params,
                name=name,
                activation=tf.keras.layers.LeakyReLU(),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                data_format="channels_last",
                kernel_initializer=initializer
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
                kernel_initializer=initializer
            )
        else:
            self.hidden_conv = None

    def call(self, yt, ht):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """

        input = yt
        for layer in self.downsampling_block:
            input = layer(input)
            summary_histograms(layer, input, layer.trainable_weights)
        for layer in self.convolution_block:
            input = layer(input)
            summary_histograms(layer, input, layer.trainable_weights)
        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(input, ht_1)  # to be recombined in new state
        summary_histograms(self.gru1, ht_1, self.gru1.trainable_weights)
        if self.hidden_conv is not None:
            ht_1_features = self.hidden_conv(ht_1)
            summary_histograms(self.hidden_conv, ht_1_features, self.hidden_conv.trainable_weigths)
        else:
            ht_1_features = ht_1
        ht_2 = self.gru2(ht_1_features, ht_2)
        summary_histograms(self.gru2, ht_2, self.gru2.trainable_weights)
        # ===========================
        delta_xt = self.upsampling_block[0](ht_2)
        summary_histograms(self.upsampling_block[0], delta_xt, self.upsampling_block[0].trainable_weights)
        for layer in self.upsampling_block[1:]:
            delta_xt = layer(delta_xt)
            summary_histograms(layer, delta_xt, layer.trainable_weights)
        for layer in self.transposed_convolution_block:
            delta_xt = layer(delta_xt)
            summary_histograms(layer, delta_xt, layer.trainable_weights)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return delta_xt, new_state


def summary_histograms(layer, activation=None, weights=None):
    if activation is not None:
        tf.summary.histogram(layer.name + "_activation", data=activation, step=tf.summary.experimental.get_step())
    if weights is not None:
        for w in weights:
            tf.summary.histogram(w.name, data=w, step=tf.summary.experimental.get_step())