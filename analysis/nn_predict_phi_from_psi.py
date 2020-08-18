import tensorflow as tf
import numpy as np
import scipy as sp
from ExoRIM.utilities import AUTOTUNE
from ExoRIM.operators import closure_phase_operator, Baselines, redundant_phase_closure_operator
from ExoRIM.physical_model import PhysicalModel
from ExoRIM.simulated_data import CenteredImagesGenerator
from ExoRIM.definitions import chisqgrad_vis, chisqgrad_amp
import os
import matplotlib.pyplot as plt

#=====================================================
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
# Modified by: Alexandre Adam

from tensorflow.keras import backend as K
import sys;sys.path.append('.')
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
from numpy.random import RandomState


# from tensorflow init_ops since it isn't available through the package
def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


class ComplexDense(Layer):
    """Regular complex densely-connected NN layer.
    `Dense` implements the operation:
    `real_preact = dot(real_input, real_kernel) - dot(imag_input, imag_kernel)`
    `imag_preact = dot(real_input, imag_kernel) + dot(imag_input, real_kernel)`
    `output = activation(K.concatenate([real_preact, imag_preact]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    AN ERROR MESSAGE IS PRINTED.
    # Arguments
        units: Positive integer, dimensionality of each of the real part
            and the imaginary part. It is actualy the number of complex units.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
            By default it is 'complex'.
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
    # Input shape
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        For a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = _compute_fans(kernel_shape)
        if self.init_criterion == 'he':
            s = np.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = np.sqrt(1. / (fan_in + fan_out))

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * K.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * K.sin(phase)"""

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return tf.Variable(initializers.random_normal(mean=0, stddev=s)(shape=shape, dtype=dtype), trainable=True)

        def init_w_imag(shape, dtype=None):
            return tf.Variable(initializers.random_normal(mean=0, stddev=s)(shape=shape, dtype=dtype), trainable=True)

        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]

        cat_kernels_4_real = K.concatenate(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = K.concatenate(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = K.dot(inputs, cat_kernels_4_complex)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PhysicalModelPhases:
    # we just want the BLM and CPO
    def __init__(self):
        B = Baselines(mask)
        self.C = tf.constant(closure_phase_operator(B), dtype)
        # self.C = tf.constant(redundant_phase_closure_operator(B), dtype)
        self.B = tf.constant(B.BLM, dtype)

    def forward(self, phi):
        return tf.einsum("ij, ...j -> ...i", self.C, phi)

    def noisy_forward(self, phi):
        # batch = phi.shape[0]
        return tf.einsum("ij, ...j -> ...i", self.C, phi) + tf.random.normal(shape=[q], mean=0, stddev=0.001)


class LinkFunction:

    def __init__(self):
        self.input_dim = 2*q
        self.output_dim = 2*p

    @staticmethod
    def forward(x):
        out = tf.concat([tf.math.cos(x), tf.math.sin(x)], axis=-1)
        return out #tf.math.sin(x) #tf.math.sin(x)

    @staticmethod
    def backward(x):
        out = tf.math.atan2(x[..., p:], x[..., :p])
        return out # tf.math.asin(x) #tf.math.asin(x)

N = 21  # apertures
L = 6
p = N * (N - 1) // 2
q = (N - 1) * (N - 2) // 2
# q = N * (N - 1) * (N - 2) // 6
pixels = 32
wavel = 0.5e-6
plate_scale = 1
SNR = 100
mask = np.random.normal(0, L, (N, 2))
x = (L + np.random.normal(0, 1, N)) * np.cos(2 * np.pi * np.arange(N) / N)
y = (L + np.random.normal(0, 1, N)) * np.sin(2 * np.pi * np.arange(N) / N)
dtype = tf.float32
mycomplex = tf.complex64
if not os.path.isdir("../logs/experiment4"):
    os.mkdir("../logs/experiment4")
logdir = "../logs/experiment4"
phy1 = PhysicalModel(pixels, mask, wavel, plate_scale, SNR, vis_phase_std=0.0001)
phy2 = PhysicalModelPhases()
link = LinkFunction()


def dataset_gen():
    gen = CenteredImagesGenerator(phy1, 1000, pixels=pixels)
    for X, Y in gen.generator():
        true_phi = tf.math.angle(phy1.forward(tf.reshape(Y, (1, *Y.shape)))[0, :p])
        phi = tf.math.angle(X[:p])
        psi = phy2.noisy_forward(phi)
        true_phi = link.forward(true_phi)
        psi = link.forward(psi)
        yield psi, true_phi  # X, Y


def test_dataset_gen():
    phy1 = PhysicalModel(pixels, mask, wavel, plate_scale, SNR, vis_phase_std=0.001)
    gen = CenteredImagesGenerator(phy1, 10, pixels=pixels)
    gen.epoch = 14159
    for X, Y in gen.generator():
        yield X, Y #psi[0], phi[0]  #


def valid_dataset_gen():
    gen = CenteredImagesGenerator(phy1, 100, pixels=pixels)
    gen.epoch = 212121
    for X, Y in gen.generator():
        true_phi = tf.math.angle(phy1.forward(tf.reshape(Y, (1, *Y.shape)))[0, :p])
        phi = tf.math.angle(X[:p])
        psi = phy2.noisy_forward(phi)
        true_phi = link.forward(true_phi)
        psi = link.forward(psi)
        yield psi, true_phi  # X, Y


def dataset(batch_size):
    dataset = tf.data.Dataset.from_generator(dataset_gen, output_types=(dtype, dtype), output_shapes=(link.input_dim, link.output_dim))
    dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    return dataset


def valid_dataset(batch_size):
    dataset = tf.data.Dataset.from_generator(valid_dataset_gen, output_types=(dtype, dtype), output_shapes=(link.input_dim, link.output_dim))
    dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    return dataset


def _model():
    m = tf.keras.Sequential(
        [
            ComplexDense(5*link.input_dim, activation="relu", name="layer1"),
            ComplexDense(3 * link.output_dim, activation="relu", name="layer2"),
            ComplexDense(link.output_dim//2, activation="tanh", name="Output")

            # [
        #     tf.keras.layers.Dense(int(5 * link.input_dim), activation="relu", name="layer1"),
        #     tf.keras.layers.Dense(int(3 * link.output_dim), activation="relu", name="layer2"),
        #     tf.keras.layers.Dense(link.output_dim, activation="tanh", name="layer3"),
        ]
    )
    return m


class SquaredAngleLoss(tf.keras.metrics.Metric):
    def __init__(self, name="Squared Angle Loss", **kwargs):
        super(SquaredAngleLoss, self).__init__(name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = link.backward(y_true)
        y_pred = link.backward(y_pred)
        self.mean.update_state(tf.square(y_true - y_pred))

    def result(self):
        return self.mean.result()

    def reset_states(self):
        self.mean.reset_states()


class AbsoluteAngleLoss(tf.keras.metrics.Metric):
    def __init__(self, name="Absolute Angle Loss", **kwargs):
        super(AbsoluteAngleLoss, self).__init__(name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = link.backward(y_true)
        y_pred = link.backward(y_pred)
        self.mean.update_state(tf.math.abs(y_true - y_pred))

    def result(self):
        return self.mean.result()

    def reset_states(self):
        self.mean.reset_states()


def main():
    model = _model()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-2, decay_rate=0.91, staircase=True, decay_steps=2000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[SquaredAngleLoss(), AbsoluteAngleLoss()])
    model.build((None, link.input_dim))
    model.summary()
    X = dataset(20)
    Valid = dataset(100)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.fit(X, epochs=100, callbacks=[tensorboard_callback], validation_data=Valid)
    image_coords = np.arange(pixels) - pixels / 2.
    xx, yy = np.meshgrid(image_coords, image_coords)
    noise = np.zeros_like(xx)
    xx_prime = xx - 9
    yy_prime = yy - 9
    rho_prime = np.sqrt(xx_prime**2 + yy_prime**2)
    noise += np.exp(-rho_prime**2/4**2)
    for X, Y in test_dataset_gen():
        y = Y.numpy().reshape((pixels**2))
        data_amp = tf.math.abs(X[:p])
        data_vis = tf.reshape(X[:p], (1, p))
        true_phi = tf.math.angle(data_vis)[0]
        data_psi = tf.math.angle(X[p:])
        data_psi = link.forward(data_psi)
        predicted_phi = model.predict(tf.reshape(data_psi, (1, link.input_dim)))[0]
        predicted_phi = link.backward(predicted_phi)
        predicted_vis = tf.complex(data_amp * tf.math.cos(predicted_phi), data_amp * tf.math.sin(predicted_phi))
        predicted_vis = tf.reshape(predicted_vis, (1, *predicted_vis.shape))
        grad_vis = chisqgrad_vis(y + noise.reshape((1, pixels**2)), phy1.A, data_vis, 1/SNR, pixels)
        predicted_grad_vis = chisqgrad_vis(y + noise.reshape((1, pixels**2)), phy1.A, predicted_vis, 1/SNR, pixels)
        fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=80)
        axs[0, 0].set_title("Ground Truth")
        im = axs[0, 0].imshow(Y.numpy().reshape((pixels, pixels)), cmap="gray")
        plt.colorbar(im, ax=axs[0, 0])
        axs[0, 1].set_title("Prediction")
        im = axs[0, 1].imshow(y.reshape((pixels, pixels)) + noise, cmap="gray")
        plt.colorbar(im, ax=axs[0, 1])
        axs[1, 0].set_title(r"$\nabla_x \chi^2_{vis}$")
        im = axs[1, 0].imshow(1/SNR**2*grad_vis.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[1, 0])
        axs[1, 1].set_title(r"Predicted $\nabla_x \chi^2_{vis}$")
        im = axs[1, 1].imshow(1/SNR**2*predicted_grad_vis.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[1, 1])
        plt.figure()
        plt.title("Raw Prediction vs real/imag of true phasors")
        plt.hist(np.abs(link.forward(predicted_phi).numpy() - link.forward(true_phi).numpy()), bins=25)
        plt.figure()
        plt.title("Phase Prediction vs truth")
        plt.hist(np.abs(predicted_phi.numpy() - true_phi.numpy()), bins=25)
        plt.figure()
        plt.title("Predicted real part of visibility")
        plt.hist(np.abs(predicted_vis.numpy().real[0] - data_vis.numpy().real[0]), bins=25)
        plt.figure()
        plt.title("Predicted imaginary part of visibility")
        plt.hist(np.abs(predicted_vis.numpy().imag[0] - data_vis.numpy().imag[0]), bins=25)

        plt.show()
        # print(f"link pred = {link.forward(predicted_phi)[0]:.3f}, gtruth = {link.forward(true_phi)[0]:.3f}")
        # print(f"link pred = {link.forward(predicted_phi)[1]:.3f}, gtruth = {link.forward(true_phi)[1]:.3f}")
        # print(f"link pred = {link.forward(predicted_phi)[2]:.3f}, gtruth = {link.forward(true_phi)[2]:.3f}")
        # print(f"link pred = {link.forward(predicted_phi)[3]:.3f}, gtruth = {link.forward(true_phi)[3]:.3f}")
        # print(f"pred = {predicted_phi[0]:.3f}, gtruth = {true_phi[0]:.3f}")
        # print(f"pred = {predicted_phi[10]:.3f}, gtruth = {true_phi[10]:.3f}")
        # print(f"pred = {predicted_phi[20]:.3f}, gtruth = {true_phi[20]:.3f}")
        # print(f"pred = {predicted_phi[200]:.3f}, gtruth = {true_phi[200]:.3f}")
        # print(f"pred = {predicted_vis.numpy().real[0, 0]:.3f} + i{predicted_vis.numpy().imag[0, 0]:.3f}  , gtruth = {data_vis.numpy().real[0, 0]:.3f} + i{data_vis.numpy().imag[0, 0]:.3f}")
        # print(f"pred = {predicted_vis.numpy().real[0, 10]:.3f} + i{predicted_vis.numpy().imag[0, 10]:.3f}  , gtruth = {data_vis.numpy().real[0, 10]:.3f} + i{data_vis.numpy().imag[0, 10]:.3f}")
        # print(f"pred = {predicted_vis.numpy().real[0, 20]:.3f} + i{predicted_vis.numpy().imag[0, 20]:.3f}  , gtruth = {data_vis.numpy().real[0, 20]:.3f} + i{data_vis.numpy().imag[0, 20]:.3f}")
        # print(f"pred = {predicted_vis.numpy().real[0, 200]:.3f} + i{predicted_vis.numpy().imag[0, 200]:.3f}  , gtruth = {data_vis.numpy().real[0, 200]:.3f} + i{data_vis.numpy().imag[0, 200]:.3f}")


if __name__ == '__main__':
    main()