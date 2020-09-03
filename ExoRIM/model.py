import tensorflow as tf
from ExoRIM.definitions import initializer, default_hyperparameters, dtype


class ConvGRU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvGRU, self).__init__(dtype=dtype, **kwargs)
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


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size, alpha, kernel_reg_amp, bias_reg_amp, **kwargs):
        super(ResidualBlock, self).__init__(dtype, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha),
            kernel_initializer=initializer,
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
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
        )

    def call(self, X):
        features = self.conv1(X)
        features = self.conv2(features)
        return tf.add(X, features)


class Model(tf.keras.models.Model):
    def __init__(self, hyperparameters=default_hyperparameters, dtype=dtype, **kwargs):
        try:
            name = hyperparameters["name"]
        except KeyError:
            name = "RIM"
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

        self.max_pool = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, yt, ht):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        # TODO remove summary or reduce number printed since this slows down enormously the training time
        input = yt
        input = self.batch_norm[0](input)
        for layer in self.downsampling_block:
            input = layer(input)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, input)
        input = self.batch_norm[1](input)
        for layer in self.convolution_block:
            input = layer(input)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, input)
        input = self.batch_norm[2](input)
        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(input, ht_1)  # to be recombined in new state
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


class Modelv2(tf.keras.models.Model):
    """
    A different version of the model, with a single GRU cell and a residual block after downsampling.
    """
    def __init__(self, hyperparameters, dtype=dtype, **kwargs):
        super(Modelv2, self).__init__(dtype, **kwargs)
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        kernel_reg_amp = hyperparameters["Regularizer Amplitude"]["kernel"]
        bias_reg_amp = hyperparameters["Regularizer Amplitude"]["bias"]
        self.conv1 = tf.keras.layers.Conv2D(
            filters=hyperparameters["Conv1"]["filters"],
            kernel_size=hyperparameters["Conv1"]["kernel_size"],
            strides=2,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha=hyperparameters["alpha"]),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            name="Conv1"
        )
        self.res1 = ResidualBlock(
            filters=hyperparameters["Res1"]["filters"],
            kernel_size=hyperparameters["Res1"]["kernel_size"],
            alpha=hyperparameters["alpha"],
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            name="Res1"
        )
        self.gru = ConvGRU(
            filters=hyperparameters["GRU"]["filters"],
            kernel_size=hyperparameters["GRU"]["kernel_size"],
            name="GRU"
        )
        self.tconv1 = tf.keras.layers.Conv2DTranspose(
            filters=hyperparameters["TConv"]["filters"],
            kernel_size=hyperparameters["TConv"]["kernel_size"],
            strides=1,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha=hyperparameters["alpha"]),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            name="TConv1"
        )
        self.fraction_tconv = tf.keras.layers.Conv2DTranspose(
            filters=hyperparameters["Fraction_TConv"]["filters"],
            kernel_size=hyperparameters["Fraction_TConv"]["kernel_size"],
            strides=2,
            padding="same",
            data_format="channels_last",
            activation=tf.keras.layers.LeakyReLU(alpha=hyperparameters["alpha"]),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            name="Fraction_TConv"
        )

    # TODO remove summaries when analysis are completed
    def call(self, xt, ht):
        features = self.conv1(xt)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.conv1, features)

        features = self.res1(features)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.res1, features)

        ht_2 = self.gru(features, ht)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru, ht_2)

        delta_xt = self.tconv1(ht_2)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.tconv1, delta_xt)

        delta_xt = self.fraction_tconv(delta_xt)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.fraction_tconv, delta_xt)
        return delta_xt, ht_2


class FeedForwardModel(tf.keras.models.Model):
    """
    Take as input the amplitude and closure phase data, and tries to decode them into an image.
    """
    def __init__(self, hparams, *args, **kwargs):
        super(FeedForwardModel, self).__init__(dtype=dtype, *args, **kwargs)
        self.dense_block = []
        self.conv_block = []
        self.upsampling_block = []
        self.batch_norm = []
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        kernel_reg_amp = hparams["Regularizer Amplitude"]["kernel"]
        bias_reg_amp = hparams["Regularizer Amplitude"]["bias"]
        try:
            batch_norm_params = hparams["Batch Norm"]
        except KeyError:
            batch_norm_params = {}
        for i in range(3):
            self.batch_norm.append(tf.keras.layers.BatchNormalization(axis=-1, **batch_norm_params))

        for layer in hparams["Dense Block"]:
            name = list(layer.keys())[0]
            self.dense_block.append(tf.keras.layers.Dense(
                name=name,
                **layer[name],
                activation=tf.keras.layers.LeakyReLU(alpha=hparams["alpha"]),
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            ))
        for layer in hparams["Conv Block"][:-1]:
            name = list(layer.keys())[0]
            self.conv_block.append(tf.keras.layers.Conv2DTranspose(
                name=name,
                **layer[name],
                padding="same",
                data_format="channels_last",
                activation=tf.keras.layers.LeakyReLU(alpha=hparams["alpha"]),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            ))
        for layer in hparams["Upsampling Block"]:
            name = list(layer.keys())[0]
            self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                name=name,
                **layer[name],
                padding="same",
                data_format="channels_last",
                activation=tf.keras.layers.LeakyReLU(alpha=hparams["alpha"]),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            ))
        self.last_layer = tf.keras.layers.Conv2D(
            filters="1",
            kernel_size=3,
            padding="same",
            data_format="channels_last",
            kernel_initializer=initializer
        )

    def call(self, X):
        X = self.batch_norm[0](X)
        for layer in self.dense_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)
        X = self.batch_norm[1](X)
        pix = int(X.shape[1]**(1/2))
        X = tf.reshape(X, shape=[X.shape[0], pix, pix, 1])
        for layer in self.upsampling_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)
        for layer in self.conv_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)
        X = self.batch_norm[2](X)
        X = self.last_layer(X)
        X = tf.keras.activations.softmax(X, axis=[1, 2])
        return X


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
