import tensorflow as tf
from exorim.models.utils import get_activation


class ConvDecodingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size=3,
            filters=32,
            conv_layers=2,
            activation="linear",
            name=None,
            strides=2,
            bilinear=False,
            **common_params
    ):
        super(ConvDecodingLayer, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides =strides
        self.activation = get_activation(activation)

        self.conv_layers = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation=self.activation,
                    **common_params
                )
            )
        if bilinear:
            self.upsampling_layer = tf.keras.layers.UpSampling2D(size=self.strides, interpolation="bilinear")
        else:
            self.upsampling_layer = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=self.activation,
                **common_params
            )

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        x = self.upsampling_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x
