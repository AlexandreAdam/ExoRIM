import tensorflow as tf
from .layers import ConvDecodingLayer, ConvEncodingLayer, ConvGRUBlock
from .utils import get_activation
from exorim.definitions import DTYPE


class Model(tf.keras.Model):
    def __init__(
            self,
            name="RIMModel",
            filters=32,
            filter_scaling=1,
            kernel_size=3,
            layers=2,
            block_conv_layers=2,
            strides=2,
            output_filters=1,
            input_kernel_size=7,
            upsampling_interpolation=False,
            activation="tanh",
            trainable=True,
            initializer="glorot_uniform",
    ):
        super(Model, self).__init__(name=name)
        self.trainable = trainable
        common_params = {
            "padding": "same",
            "kernel_initializer": initializer,
            "data_format": "channels_last"
        }

        self.state_depth = 2*int(filter_scaling**layers * filters)
        self.downsampling_factor = strides**layers
        activation = get_activation(activation)

        self._num_layers = layers
        self._strides = strides
        self._init_filters = filters
        self._filter_scaling = filter_scaling

        self.encoding_layers = []
        self.decoding_layers = []
        for i in range(layers):
            self.encoding_layers.append(
                ConvEncodingLayer(
                    kernel_size=kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    downsampling_filters=int(filter_scaling ** (i + 1) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    **common_params
                )
            )
            self.decoding_layers.append(
                ConvDecodingLayer(
                    kernel_size=kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    bilinear=upsampling_interpolation,
                    **common_params
                )
            )

        self.decoding_layers = self.decoding_layers[::-1]

        self.bottleneck_gru = ConvGRUBlock(
            filters=int(filter_scaling**layers * filters),
            kernel_size=kernel_size,
            activation=activation
        )

        self.input_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=input_kernel_size,
            activation=activation,
            **common_params
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            activation="linear",
            **common_params
        )

    def __call__(self, delta_xt, states):
        return self.call(delta_xt, states)

    def call(self, delta_xt, states):
        delta_xt = self.input_layer(delta_xt)
        for i in range(len(self.encoding_layers)):
            delta_xt = self.encoding_layers[i](delta_xt)
        delta_xt, new_state = self.bottleneck_gru(delta_xt, states)
        for i in range(len(self.decoding_layers)):
            delta_xt = self.decoding_layers[i](delta_xt)
        delta_xt = self.output_layer(delta_xt)
        return delta_xt, new_state

    def init_hidden_states(self, input_pixels, batch_size):
        state_size = input_pixels // self.downsampling_factor
        return tf.zeros(shape=(batch_size, state_size, state_size, self.state_depth), dtype=DTYPE)
