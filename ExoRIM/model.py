import tensorflow as tf
import numpy as np
from ExoRIM.definitions import dtype, default_hyperparameters
from ExoRIM.kpi import kpi
from ExoRIM.utilities import save_output
import time
import os


class ConvGRU(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ConvGRU, self).__init__()
        self.update_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='sigmoid',
            padding='same'
        )
        self.reset_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='sigmoid',
            padding='same'
        )
        self.candidate_activation_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation='tanh',
            padding='same'
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
                data_format="channels_last"
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
                data_format="channels_last"
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
                data_format="channels_last"
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
                data_format="channels_last"
            ))
        self.gru1 = ConvGRU(**hyperparameters["Recurrent Block"]["GRU_1"])
        self.gru2 = ConvGRU(**hyperparameters["Recurrent Block"]["GRU_2"])
        self.hidden_conv = tf.keras.layers.Conv2DTranspose(
            **hyperparameters["Recurrent Block"]["Hidden_Conv_1"],
            name="Hidden_Conv_1",
            activation=tf.keras.layers.LeakyReLU(),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            data_format="channels_last"
        )

    def call(self, xt, ht, grad):
        """
        :param inputs: List = [xt, ht, grad]
            xt: Image tensor of shape (batch size, num of pixel, num of pixel, channels)
            ht: Hidden state list: [h^0_t, h^1_t, ...]
                    (batch size, downsampled image size, downsampled image size, state_size * num of ConvGRU cells)
            grad: Gradient of the log-likelihood function for y (data) given xt
        :return: x_{t+1}, h_{t+1}
        """
        stacked_input = tf.concat([xt, grad], axis=3)
        for layer in self.downsampling_block:
            stacked_input = layer(stacked_input)
        for layer in self.convolution_block:
            stacked_input = layer(stacked_input)
        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(stacked_input, ht_1)  # to be recombined in new state
        ht_1_features = self.hidden_conv(ht_1)
        ht_2 = self.gru2(ht_1_features, ht_2)
        # ===========================
        delta_xt = self.upsampling_block[0](ht_2)
        for layer in self.upsampling_block[1:]:
            delta_xt = layer(delta_xt)
        for layer in self.transposed_convolution_block:
            delta_xt = layer(delta_xt)
        xt_1 = xt + delta_xt
        xt_1 = tf.sigmoid(xt_1)  # Link function to normalize output
        # softmax or sigmoid
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return xt_1, new_state


class RIM:
    def __init__(self, mask_coordinates, hyperparameters, dtype=dtype, weight_file=None):
        self._dtype = dtype
        self.hyperparameters = hyperparameters
        self.channels = hyperparameters["channels"]
        self.pixels = hyperparameters["pixels"]
        self.steps = hyperparameters["steps"]
        self.state_size = hyperparameters["state_size"]
        self.state_depth = hyperparameters["state_depth"]
        self.model = Model(hyperparameters, dtype=self._dtype)
        if weight_file is not None:
            self.model.load_weights(weight_file)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)  # Setting for channels last
        self.physical_model = PhysicalModel(
            mask_coordinates=mask_coordinates,
            pixels=self.pixels,
            visibility_noise=hyperparameters["Physical Model"]["Visibility Noise"],
            cp_noise=hyperparameters["Physical Model"]["Closure Phase Noise"]
        )

    def call(self, X):
        """
        Method used in training to get model predictions.

        :param X: Vector of complex visibilities amplitude and closure phases
        :return: 5D Tensor of shape (batch_size, [image_size, channels], steps)
        """
        batch_size = X.shape[0]
        x0 = self.initial_guess(batch_size)
        h0 = self.init_hidden_states(batch_size)
        # Compute the gradient through auto-diff since model is highly non-linear
        with tf.GradientTape() as g:
            g.watch(x0)
            likelihood = self.log_likelihood(x0, X)
        grad = self.batch_norm(g.gradient(likelihood, x0), training=True)
        xt, ht = self.model(x0, h0, grad)
        outputs = tf.reshape(xt, xt.shape + [1])  # Plus one dimension for step stack
        for current_step in range(self.steps - 1):
            with tf.GradientTape() as g:
                g.watch(xt)
                likelihood = self.log_likelihood(xt, X)
            grad = self.batch_norm(g.gradient(likelihood, xt), training=True)
            xt, ht = self.model(xt, ht, grad)
            outputs = tf.concat([outputs, tf.reshape(xt, xt   .shape + [1])], axis=4)
        # sigmoid --> prevent negative pixels
        return outputs

    def predict(self, X):

        """
        Returns the reconstructed images from interferometric data.

        :param X: Vector of complex visibilities amplitude and closure phases
        :return: 4D Tensor of shape (batch_size, [image_size, channels])
        """
        batch_size = X.shape[0]
        x0 = self.initial_guess(batch_size)
        h0 = self.init_hidden_states(batch_size)
        # Compute the gradient through auto-diff since model is highly non-linear
        with tf.GradientTape() as g:
            g.watch(x0)
            likelihood = self.log_likelihood(x0, X)
        grad = self.batch_norm(g.gradient(likelihood, x0), training=False)  # freeze the learning of Batch Norm
        xt, ht = self.model(x0, h0, grad)
        outputs = tf.reshape(xt, xt.shape + [1])  # Plus one dimension for step stack
        for current_step in range(self.steps - 1):
            with tf.GradientTape() as g:
                g.watch(xt)
                likelihood = self.log_likelihood(xt, X)
            grad = self.batch_norm(g.gradient(likelihood, xt), training=False)
            xt, ht = self.model(xt, ht, grad)
            outputs = tf.concat([outputs, tf.reshape(xt, xt.shape + [1])], axis=4)
        return outputs

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.state_size, self.state_size, self.state_depth), dtype=self._dtype)

    def initial_guess(self, batch_size):
        # Initial guess cannot be zeros, it blows up the gradient of the log-likelihood
        x0 = tf.zeros(shape=(batch_size, self.pixels, self.pixels, self.channels), dtype=self._dtype) + 1e-4
        return x0

    def log_likelihood(self, xt, y):
        """
        Mean squared error of the reconstructed yhat vector and the true y vector, divided by the inverse of the
        covariance matrix (diagonal in our case).

        :param y: True vector of complex visibility amplitudes and closure phases
        :param xt: Image reconstructed at step t of the reconstruction
        :return: Scalar L
        """
        yhat = self.physical_model.physical_model(xt)
        return 0.5 * tf.math.reduce_sum(tf.square(y - yhat)) / self.physical_model.error_tensor**2

    def fit(
            self,
            train_dataset,
            cost_function,
            max_time,
            patience=10,
            checkpoints=5,
            min_delta=0,
            max_epochs=1000,
            test_dataset=None,
            output_dir=None,
            output_save_mod=50,
            checkpoint_dir=None
    ):
        """

        :param train_dataset:
        :param cost_function:
        :param max_time: Maximum time allowed for training in hours
        :param patience:
        :param min_delta:
        :param max_epochs:
        :param test_dataset:
        :return: loss_history
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        start = time.time()
        epoch = 1
        history = {"train_loss": [], "test_loss": []}
        min_loss = np.inf
        epoch_loss = tf.metrics.Mean()
        _patience = patience
        while _patience > 0 and epoch < max_epochs and (time.time() - start) < max_time*3600:
            epoch_loss.reset_states()
            for batch, (X, Y) in train_dataset:  # X and Y by ML convention, batch is an index
                batch = batch.numpy()
                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_weights)
                    output = self.call(X)
                    cost_value = cost_function(output, Y)
                    cost_value += tf.reduce_sum(self.model.losses)  # Add layer specific regularizer losses (L2 in definitions)
                epoch_loss.update_state([cost_value])
                gradient = tape.gradient(cost_value, self.model.trainable_weights)
                clipped_gradient, _ = tf.clip_by_global_norm(gradient, clip_norm=10)  # prevent exploding gradients
                optimizer.apply_gradients(zip(clipped_gradient, self.model.trainable_weights))
                if output_dir is not None:
                    save_output(output, output_dir, epoch, batch, output_save_mod)
            history["train_loss"].append(epoch_loss.result().numpy())
            if test_dataset is not None:
                for X, Y in test_dataset:
                    test_output = self.predict(X)
                    test_cost = cost_function(test_output, Y)
                    test_cost += tf.reduce_sum(self.model.losses)
                    history["test_loss"].append([test_cost])

            if checkpoint_dir is not None:
                if epoch % checkpoints == 0:
                    self.model.save_weights(os.path.join(checkpoint_dir, f"rim_{epoch:03}_{cost_value:.5f}.h5"))
            if cost_value < min_loss - min_delta:
                _patience = patience
                min_loss = cost_value
            else:
                _patience -= 1
            epoch += 1
        if "epoch" in self.hyperparameters.keys():
            self.hyperparameters["epoch"] += epoch
        else:
            self.hyperparameters["epoch"] = epoch
        return history


class CostFunction(tf.keras.losses.Loss):
    def __init__(self):
        super(CostFunction, self).__init__()
        # Possible framework for weights
        # if cost_weights is None:
        #     self.cost_weights = [1] * steps
        # elif not isinstance(cost_weights, list):
        #     raise TypeError("cost_weights must be a list")
        # elif len(cost_weights) != steps:
        #     raise IndexError("cost_weights must be a list of length steps ")

    def call(self, X, Y):
        """
        Dimensions are
        0: batch_size
        1: x dimension of image
        2: y dimension of image
        3: channels
        4: time steps of the RIM output
        :param x_true: 4D tensor to be compared with x_preds
        :param x_preds: 5D tensor output of the call method
        :return:
        """ # TODO MSE on log of pixels
        batch_size = X.shape[0]
        pixels = X.shape[1] * X.shape[2] * X.shape[3]
        steps = X.shape[4]
        Y_ = tf.reshape(Y, (Y.shape + [1]))
        cost = tf.reduce_sum(tf.square(Y_ - X), axis=[1, 2, 3])/pixels  # Sum over pixels
        cost = tf.reduce_sum(cost, axis=1)/steps  # Sum over time steps (this one could be weighted)
        cost = tf.reduce_sum(cost, axis=0)/batch_size  # Mean over the batch
        return cost


class PhysicalModel(object):

    def __init__(self, mask_coordinates, pixels, visibility_noise, cp_noise):
        """
        :param coord_file: Numpy array with coordinates of holes in non-redundant mask (in meters)
        :param pixels: Number of pixel on the side of a square camera
        :param visibility_noise: Standard deviation of the visibilty amplitude squared
        :param cp_noise: Standard deviation of the closure phases
        """
        print('initializing Phys_Mod')
        self.pixels = pixels
        self.visibility_noise = visibility_noise
        self.cp_noise = cp_noise
        bs = kpi(mask=mask_coordinates, bsp_mat='sparse')

        ## create p2vm matrix

        x = np.arange(self.pixels)
        xx, yy = np.meshgrid(x, x)

        p2vm_sin = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        #TODO change this for FFT, might be too slow
        for j in range(bs.uv.shape[0]):
            p2vm_sin[j, :] = np.ravel(np.sin(2*np.pi*(xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

        p2vm_cos = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        for j in range(bs.uv.shape[0]):
            p2vm_cos[j, :] = np.ravel(np.cos(2*np.pi*(xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

        # create tensor to hold cosine and sine projection operators
        self.cos_projector = tf.constant(p2vm_cos.T, dtype=dtype)
        self.sin_projector = tf.constant(p2vm_sin.T, dtype=dtype)
        self.bispectra_projector = tf.constant(bs.uv_to_bsp.T, dtype=dtype)

        vis2s = np.zeros(p2vm_cos.shape[0])
        closure_phases = np.zeros(bs.uv_to_bsp.shape[0])
        self.vis2s_tensor = tf.constant(vis2s, dtype=dtype)
        self.cp_tensor = tf.constant(closure_phases, dtype=dtype)
        self.data_tensor = tf.concat([self.vis2s_tensor, self.cp_tensor], 0)

        # create tensor to hold your uncertainties
        self.vis2s_error = tf.constant(np.ones_like(vis2s) * self.visibility_noise, dtype=dtype)
        self.cp_error = tf.constant(np.ones_like(closure_phases) * self.cp_noise, dtype=dtype)
        self.error_tensor = tf.concat([self.vis2s_error, self.cp_error], axis=0)

    def physical_model(self, image):
        flat = tf.keras.layers.Flatten(data_format="channels_last")(image)
        sin_model = tf.tensordot(flat, self.sin_projector, axes=1)
        cos_model = tf.tensordot(flat, self.cos_projector, axes=1)
        visibility_squared = tf.square(cos_model) + tf.square(sin_model)
        phases = tf.math.angle(tf.complex(cos_model, sin_model))
        closure_phase = tf.tensordot(phases, self.bispectra_projector, axes=1)
        y = tf.concat([visibility_squared, closure_phase], axis=1)
        return y

    def simulate_noisy_image(self, image):
        # apply noise to image before passing in physical model
        out = self.physical_model(image)
        out += self.error_tensor
        return out
