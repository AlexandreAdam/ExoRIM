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
    def __init__(self, log_lieklihood, hyperparameters, dtype=dtype, weight_file=None, arrays=None):
        self._dtype = dtype
        self.hyperparameters = hyperparameters
        self.channels = hyperparameters["channels"]
        self.pixels = hyperparameters["pixels"]
        self.steps = hyperparameters["steps"]
        self.state_size = hyperparameters["state_size"]
        self.state_depth = hyperparameters["state_depth"]
        self.model = Model(hyperparameters, dtype=self._dtype)
        if weight_file is not None:
            x = self.initial_guess(1)
            h = self.init_hidden_states(1)
            self.model.call(x, h, x)
            self.model.load_weights(weight_file)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)  # Setting for channels last
        self.log_likelihood = chi_squared

    def __call__(self, X, training=False): # used for prediction
        return self.call(X, training)

    def call(self, X, training=True): # used for training
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
        grad = self.batch_norm(g.gradient(likelihood, x0), training=training)
        xt, ht = self.model(x0, h0, grad)
        outputs = tf.reshape(xt, xt.shape + [1])  # Plus one dimension for step stack
        for current_step in range(self.steps - 1):
            with tf.GradientTape() as g:
                g.watch(xt)
                likelihood = self.log_likelihood(xt, X)
            grad = self.batch_norm(g.gradient(likelihood, xt), training=training)
            xt, ht = self.model(xt, ht, grad)
            outputs = tf.concat([outputs, tf.reshape(xt, xt.shape + [1])], axis=4)
        return outputs

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.state_size, self.state_size, self.state_depth), dtype=self._dtype)

    def initial_guess(self, batch_size):
        # Initial guess cannot be zeros, it blows up the gradient of the log-likelihood
        x0 = tf.zeros(shape=(batch_size, self.pixels, self.pixels, self.channels), dtype=self._dtype) + 1e-4
        return x0

    def fit(
            self,
            train_dataset,
            cost_function,
            optimizer,
            max_time,
            metrics=None,
            patience=10,
            track="train_loss",
            checkpoints=5,
            min_delta=0,
            max_epochs=1000,
            test_dataset=None,
            output_dir=None,
            output_save_mod=None,
            checkpoint_dir=None,
            name="rim"
    ):
        """
        This function trains the weights of the model on the training dataset, which should be a
        tensorflow Dataset class created from zipping X and Y tensors into one dataset. This dataset should have
        been initialised with the enumerate method and batch method. For better performance, use chache and prefetch.
        :param train_dataset: A tensorflow dataset created from zipping X and Y, initialised with batch and enumerate(start=0)
        :param cost_function: target for the training
        :param max_time: Time allowed for training in hours
        :param metrics: A dictionary with metric functions as items that take (Y_pred, Y_true) as input
        :param patience: An integer, the maximum number of epoch without improvement before terminating fit
        :param checkpoints: An integer, correspond to the mod of the epoch at which point the weights are saved
        :param min_delta: This compares loss between epochs; it judges if the loss has been improved if loss - previous_loss <min_delta
        :param max_epochs: Maximum number of epochs allowed
        :param test_dataset: A tensorflow dataset created from zipping X_test and Y_test. Should not be batched (or batch_size = num_samples)
        :param output_dir: directory to store output of the model
        :param output_save_mod: dictionary in the form
            output_save_mod = {
                "index_mod": 25,
                "epoch_mod": 1,
                "step_mod": 1
            },
            each entry correspond to a particula output to be saved. This examples saves the output at
            each epoch and saves each steps, but ouly saves 1 image per 25 in the dataset. This particular
            examples can produces an enourmous amount of pictures if the user is not careful.

        :param checkpoint_dir: Directory where to save the weights
        :param name: Name of the model
        :return: history, a dictionary with loss and metrics score at each epoch
        """
        if metrics is None:
            metrics = {}
        if output_save_mod is None and output_dir is not None:
            output_save_mod = {
                "index_mod": 1,
                "epoch_mod": 1,
                "step_mod": 1
            },
        start = time.time()
        if "epoch" in self.hyperparameters.keys():
            _epoch_start = self.hyperparameters["epoch"]
        else:
            _epoch_start = 0
        epoch = 1
        history = {"train_loss": [], "test_loss": []}
        history.update({key + "_train": [] for key in metrics.keys()})
        history.update({key + "_test": [] for key in metrics.keys()})
        min_score = np.inf
        epoch_loss = tf.metrics.Mean()
        _patience = patience
        while _patience > 0 and epoch < max_epochs and (time.time() - start) < max_time*3600:
            epoch_loss.reset_states()
            metrics_train = {key: 0 for key in metrics.keys()}
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
                    save_output(output, output_dir, epoch + _epoch_start, batch, **output_save_mod)
                for key, item in metrics_train.items():
                    metrics_train[key] += tf.math.reduce_mean(metrics[key](output[..., -1], Y)).numpy()
            for key, item in metrics_train.items():
                history[key + "_train"].append(item/(batch + 1))
            history["train_loss"].append(epoch_loss.result().numpy())
            if test_dataset is not None:
                for X, Y in test_dataset: # this dataset should not be batched, so this for loop has 1 iteration
                    test_output = self.call(X, training=True)  # investigate why predict returns NaN scores and output
                    test_cost = cost_function(test_output, Y)
                    test_cost += tf.reduce_sum(self.model.losses)
                    history["test_loss"].append(test_cost.numpy())
                    for key, item in metrics.items():
                        history[key + "_test"].append(tf.math.reduce_mean(item(test_output[..., -1], Y)).numpy())

            if history[track][-1] < min_score - min_delta:
                _patience = patience
                min_score = history[track][-1]
            else:
                _patience -= 1
            if checkpoint_dir is not None:
                if epoch % checkpoints == 0 or _patience == 0 or epoch == max_epochs - 1:
                    self.model.save_weights(os.path.join(checkpoint_dir, f"{name}_{epoch + _epoch_start:03}_{cost_value:.5f}.h5"))
            epoch += 1
        self.hyperparameters["epoch"] = epoch + _epoch_start
        return history


class MSE(tf.keras.losses.Loss):
    def __init__(self):
        super(MSE, self).__init__()
        # Possible framework for weights
        # if cost_weights is None:
        #     self.cost_weights = [1] * steps
        # elif not isinstance(cost_weights, list):
        #     raise TypeError("cost_weights must be a list")
        # elif len(cost_weights) != steps:
        #     raise IndexError("cost_weights must be a list of length steps ")

    def call(self, Y_pred, Y_true):
        """
        Dimensions are
        0: batch_size
        1: x dimension of image
        2: y dimension of image
        3: channels
        4: time steps of the RIM output
        :param Y_true: 4D tensor to be compared with Y_preds
        :param Y_preds: 5D tensor output of the call method
        :return: Score
        """
        Y_ = tf.reshape(Y_true, (Y_true.shape + [1]))
        cost = tf.reduce_mean(tf.square(Y_ - Y_pred), axis=[1, 2, 3])
        cost = tf.reduce_mean(cost, axis=1)
        cost = tf.reduce_mean(cost, axis=0)
        return cost


class SSIM_loss(tf.keras.losses.Loss):
    # Adapted from keras-contrib team
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5
    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.
    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        super(SSIM_loss, self).__init__()
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value

    def call(self, y_pred, y_true):
        shape = list(tf.shape(y_pred))
        steps = shape[-1]
        batch_size = shape[0]
        score = np.zeros(batch_size)
        for s in range(steps):
            score += 1.0 - tf.image.ssim(y_pred[..., s], y_true,
                                         max_val=self.max_value,
                                         filter_size=self.kernel_size,
                                         k1=self.k1,
                                         k2=self.k2).numpy()
        score /= steps
        return score.mean()


class PhysicalModel(object):

    def __init__(self, mask_coordinates, pixels, visibility_noise, cp_noise, arrays=None):
        """
        :param mask_coordinates: Numpy array with coordinates of holes in non-redundant mask (in meters)
        :param pixels: Number of pixel on the side of a square camera
        :param visibility_noise: Standard deviation of the visibilty amplitude squared
        :param cp_noise: Standard deviation of the closure phases
        :param arrays: initialize from saved tensor instead of computing kpi.
        """
        self.pixels = pixels
        self.visibility_noise = visibility_noise
        self.cp_noise = cp_noise
        if arrays is not None:
            self.cos_projector = tf.constant(arrays["cos_projector"], dtype=dtype)
            self.sin_projector = tf.constant(arrays["sin_projector"], dtype=dtype)
            self.bispectra_projector = tf.constant(arrays["bispectra_projector"], dtype=dtype)
        else:
            print('initializing Phys_Mod')
            bs = kpi(mask=mask_coordinates, bsp_mat='sparse')

            ## create p2vm matrix

            x = np.arange(self.pixels)
            xx, yy = np.meshgrid(x, x)

            p2vm_sin = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

            for j in range(bs.uv.shape[0]):
                p2vm_sin[j, :] = np.ravel(np.sin(2*np.pi*(xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

            p2vm_cos = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

            for j in range(bs.uv.shape[0]):
                p2vm_cos[j, :] = np.ravel(np.cos(2*np.pi*(xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

            # create tensor to hold cosine and sine projection operators
            self.cos_projector = tf.constant(p2vm_cos.T, dtype=dtype)
            self.sin_projector = tf.constant(p2vm_sin.T, dtype=dtype)
            self.bispectra_projector = tf.constant(bs.uv_to_bsp.T, dtype=dtype)

        vis2s = np.zeros(self.cos_projector.shape[1])
        closure_phases = np.zeros(self.bispectra_projector.shape[1])
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


class PhysicalModelv2:
    """
    Physical model based one sided Fourier Transform and closure phase operator computed from kpi.BLM (xara)
    Example of use:
        kpi = xara.kpi.KPI(mask_coordinates) # (x, y) in meters
        cpo = ExoRIM.closure_phases.closure_phase_operator(kpi)
        dftm = ExoRIM.definitions.one_sided_DFTM(kpi.UVC, wavelength, pixels, plate_scale, inv=False, dprec=True):
        phys = PhysicalModelv2(pixels, cpo, dftm)
    """
    def __init__(self, pixels, flux, bispectrum_projector, one_sided_DFTM):
        """

        :param pixels: Number of pixels on the side of the reconstructed image
        :param bispectrum_projector: Projection operator applied on the visibility to get bispectrum
        :param one_sided_DFTM: Fourier transform operator to get visibility in terms of unique baselines
        """
        self.pixels = pixels
        self.BP = tf.constant(bispectrum_projector, dtype=dtype)
        self.DFTM = tf.constant(one_sided_DFTM, dtype=tf.complex128)
        self.p = one_sided_DFTM.shape[0]  # number of visibility samples
        self.q = bispectrum_projector.shape[1]  # number of independant closure phases

    def forward(self, image, flux):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size): used to normalize complex visibilities
        :return: A concatenation of complex visibilities and bispectra (dtype: tfcomplex128)
        """
        visibilities = self.fourier_transform(image)
        bispectrum = tf.tensordot(self.BP, visibilities, axes=1) # math.angles returns type float
        y = tf.concat([visibilities, bispectrum], axis=1)  # p + q length vectors
        return y

    def fourier_transform(self, image):
        im = tf.cast.dtypes(image, tf.complex128)
        flat = tf.keras.layers.Flatten(data_format="channel_last")(im)
        return tf.tensordot(self.DFTM, flat, axes=1)


class NoiseModel:
    """

    """
    def __init__(self):
        """"""
        pass


class LogLikelihood:
    """

    """
    def __init__(self, chi_squared, regulariser):
        """"""
        pass