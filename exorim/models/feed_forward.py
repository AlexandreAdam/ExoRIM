import tensorflow as tf
from exorim.models.utils import global_step, summary_histograms
from exorim.definitions import DTYPE


class FeedForwardModel(tf.keras.models.Model):
    """
    Take as input the amplitude and closure phase data, and tries to decode them into an image.
    """
    def __init__(self, hparams, *args, **kwargs):
        super(FeedForwardModel, self).__init__(dtype=DTYPE, *args, **kwargs)
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
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
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
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            ))
        self.last_layer = tf.keras.layers.Conv2D(
            filters="1",
            kernel_size=3,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform()
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