import tensorflow as tf
from .layers import UnetDecodingLayer, UnetEncodingLayer, ConvGRUBlock
from .utils import get_activation
from exorim.definitions import DTYPE


class UnetModel(tf.keras.Model):
    def __init__(
            self,
            name="RIMUnetModel",
            filters=32,
            filter_scaling=1,
            kernel_size=3,
            layers=2,                        # before bottleneck
            block_conv_layers=2,
            strides=2,
            output_filters=1,
            bottleneck_kernel_size=None,     # use kernel_size as default
            bottleneck_filters=None,
            resampling_kernel_size=None,
            gru_kernel_size=None,
            upsampling_interpolation=False,  # use strided transposed convolution if false
            kernel_regularizer_amp=0.,
            bias_regularizer_amp=0.,
            activation="leaky_relu",
            alpha=0.1,                       # for leaky relu
            use_bias=True,
            trainable=True,
            initializer="glorot_uniform",
    ):
        super(UnetModel, self).__init__(name=name)
        self.trainable = trainable

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_regularizer_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L2(l2=bias_regularizer_amp)})

        resampling_kernel_size = resampling_kernel_size if resampling_kernel_size is not None else kernel_size
        bottleneck_kernel_size = bottleneck_kernel_size if bottleneck_kernel_size is not None else kernel_size
        bottleneck_filters = bottleneck_filters if bottleneck_filters is not None else int(filter_scaling ** (layers + 1) * filters)
        self.downsampling_factor = layers**strides
        gru_kernel_size = gru_kernel_size if gru_kernel_size is not None else kernel_size
        activation = get_activation(activation, alpha=alpha)

        self._num_layers = layers
        self._strides = strides
        self._init_filters = filters
        self._filter_scaling = filter_scaling
        self._bottleneck_filters = bottleneck_filters

        self.encoding_layers = []
        self.decoding_layers = []
        self.gated_recurrent_blocks = []
        for i in range(layers):
            self.encoding_layers.append(
                UnetEncodingLayer(
                    kernel_size=kernel_size,
                    downsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    **common_params
                )
            )
            self.decoding_layers.append(
                UnetDecodingLayer(
                    kernel_size=kernel_size,
                    upsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    bilinear=upsampling_interpolation,
                    **common_params
                )
            )
            self.gated_recurrent_blocks.append(
                    ConvGRUBlock(
                        filters=2*int(filter_scaling**(i) * filters),
                        kernel_size=gru_kernel_size,
                        activation=activation
                )
            )

        self.decoding_layers = self.decoding_layers[::-1]

        self.bottleneck_layer1 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation,
            **common_params
        )
        self.bottleneck_layer2 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation,
            **common_params
        )
        self.bottleneck_gru = ConvGRUBlock(
            filters=2*bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

    def __call__(self, xt, states, grad):
        return self.call(xt, states, grad)

    def call(self, xt, states, grad):
        delta_xt = tf.concat([tf.identity(xt), grad], axis=3)
        skip_connections = []
        for i in range(len(self.encoding_layers)):
            c_i, delta_xt = self.encoding_layers[i](delta_xt)
            skip_connections.append(c_i)
        delta_xt = self.bottleneck_layer1(delta_xt)
        delta_xt = self.bottleneck_layer2(delta_xt)
        # Pass skip connections through GRUS and update states
        new_states = []
        for i in range(len(self.gated_recurrent_blocks)):
            c_i, new_state = self.gated_recurrent_blocks[i](skip_connections[i], states[i])
            skip_connections[i] = c_i
            new_states.append(new_state)
        skip_connections = skip_connections[::-1]
        delta_xt, new_state = self.bottleneck_gru(delta_xt, states[-1])
        new_states.append(new_state)
        for i in range(len(self.decoding_layers)):
            delta_xt = self.decoding_layers[i](delta_xt, skip_connections[i])
        delta_xt = self.output_layer(delta_xt)

        xt_1 = xt + delta_xt  # update image
        return xt_1, new_states

    def init_hidden_states(self, input_pixels, batch_size, constant=0.):
        hidden_states = []
        for i in range(self._num_layers):
            pixels = input_pixels // self._strides**(i)
            filters = int(self._filter_scaling**(i) * self._init_filters)
            hidden_states.append(
                constant * tf.ones(shape=[batch_size, pixels, pixels, 2 * filters], dtype=DTYPE)
            )
        pixels = input_pixels // self._strides ** (self._num_layers)
        hidden_states.append(
            constant * tf.ones(shape=[batch_size, pixels, pixels, 2 * self._bottleneck_filters], dtype=DTYPE)
        )
        return hidden_states