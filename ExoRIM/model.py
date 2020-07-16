import tensorflow as tf
import numpy as np
from ExoRIM.definitions import dtype, default_hyperparameters, mycomplex, chisqgrad_vis, chisqgrad_bs, chisqgrad_cphase, \
    chisqgrad_amp, initializer, softmax_min_max_scaler
from ExoRIM.kpi import kpi
from ExoRIM.utilities import save_output, save_gradient_and_weights, save_loglikelihood_grad
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

    def call(self, xt, ht):
        """
        :param inputs: List = [xt, ht, grad]
            xt: Image tensor of shape (batch size, num of pixel, num of pixel, channels)
            ht: Hidden state list: [h^0_t, h^1_t, ...]
                    (batch size, downsampled image size, downsampled image size, state_size * num of ConvGRU cells)
            grad: Gradient of the log-likelihood function for y (data) given xt
        :return: x_{t+1}, h_{t+1}
        """
        input = xt
        for layer in self.downsampling_block:
            input = layer(input)
        for layer in self.convolution_block:
            input = layer(input)
        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(input, ht_1)  # to be recombined in new state
        if self.hidden_conv is not None:
            ht_1_features = self.hidden_conv(ht_1)
        else:
            ht_1_features = ht_1
        ht_2 = self.gru2(ht_1_features, ht_2)
        # ===========================
        delta_xt = self.upsampling_block[0](ht_2)
        for layer in self.upsampling_block[1:]:
            delta_xt = layer(delta_xt)
        for layer in self.transposed_convolution_block:
            delta_xt = layer(delta_xt)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return delta_xt, new_state


class RIM:
    def __init__(self, physical_model, hyperparameters, dtype=dtype, weight_file=None):
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
        self.physical_model = physical_model

    def __call__(self, X, training=False): # used for prediction
        return self.call(X, training)

    def call(self, X, training=True): # used for training
        """
        Method used in training to get model predictions.

        :param X: Vector of complex visibilities amplitude and closure phases
        :return: 5D Tensor of shape (batch_size, [image_size, channels], steps)
        """
        batch_size = X.shape[0]
        y0 = self.initial_guess(X)
        h0 = self.init_hidden_states(batch_size)
        grad = self.physical_model.grad_log_likelihood_v2(y0, X)
        # scale in range [-1, 1] to accelerate learning
        eta_0 = softmax_min_max_scaler(y0, minimum=-1., maximum=1.)
        # Scale gradient to the same dynamical range as y
        grad = softmax_min_max_scaler(grad, minimum=-1., maximum=1.)
        stacked_input = tf.concat([eta_0, grad], axis=3)
        # compute gradient update
        gt, ht = self.model(stacked_input, h0)
        # update image
        eta_t = eta_0 + gt
        # apply inverse link function to go back to image space
        yt = softmax_min_max_scaler(eta_t, minimum=0, maximum=1.)
        # save output and log likelihood gradient (latter for analysis)
        outputs = tf.reshape(yt, yt.shape + [1])  # Plus one dimension for step stack
        grads = grad
        for current_step in range(self.steps - 1):
            grad = self.physical_model.grad_log_likelihood_v2(yt, X)
            grad = softmax_min_max_scaler(grad, minimum=-1., maximum=1.)
            stacked_input = tf.concat([eta_t, grad], axis=3)
            gt, ht = self.model(stacked_input, ht)
            eta_t = eta_t + gt
            yt = softmax_min_max_scaler(eta_t, minimum=0, maximum=1.)
            outputs = tf.concat([outputs, tf.reshape(yt, yt.shape + [1])], axis=4)
            grads = tf.concat([grads, grad], axis=3)
        return outputs, grads

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.state_size, self.state_size, self.state_depth), dtype=self._dtype)

    def initial_guess(self, X):
        y0 = self.physical_model.inverse_fourier_transform(X)
        y0 = softmax_min_max_scaler(y0, minimum=0, maximum=1.)
        # y0 = tf.ones(shape=[X.shape[0], self.pixels, self.pixels, 1]) / 100
        return y0

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
                    output, grads = self.call(X)
                    cost_value = cost_function(output, Y)
                    cost_value += tf.reduce_sum(self.model.losses)  # Add layer specific regularizer losses (L2 in definitions)
                epoch_loss.update_state([cost_value])
                gradient = tape.gradient(cost_value, self.model.trainable_weights)
                clipped_gradient = gradient
                optimizer.apply_gradients(zip(clipped_gradient, self.model.trainable_weights))
                if output_dir is not None:
                    save_output(output, output_dir, epoch + _epoch_start, batch, **output_save_mod, format="txt")
                    save_gradient_and_weights(gradient, self.model.trainable_weights, output_dir, epoch + _epoch_start, batch)
                    save_loglikelihood_grad(grads, output_dir, epoch + _epoch_start, batch, **output_save_mod)
                for key, item in metrics_train.items():
                    metrics_train[key] += tf.math.reduce_mean(metrics[key](output[..., -1], Y)).numpy()
            for key, item in metrics_train.items():
                history[key + "_train"].append(item/(batch + 1))
            history["train_loss"].append(epoch_loss.result().numpy())
            if test_dataset is not None:
                for X, Y in test_dataset: # this dataset should not be batched, so this for loop has 1 iteration
                    test_output, _ = self.call(X, training=True)  # investigate why predict returns NaN scores and output
                    test_cost = cost_function(test_output, Y)
                    test_cost += tf.reduce_sum(self.model.losses)
                    history["test_loss"].append(test_cost.numpy())
                    for key, item in metrics.items():
                        history[key + "_test"].append(tf.math.reduce_mean(item(test_output[..., -1], Y)).numpy())
            print(f"{epoch}: train_loss={history['train_loss'][-1]:.2e} | val_loss={history['test_loss'][-1]:.2e}")
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

    def call(self, Y_pred, Y_true):
        """
        :param Y_true: 4D tensor to be compared with Y_preds: Has to be normalize by flux!!
        :param Y_preds: 5D tensor output of the call method
        :return: Score
        """
        Y_ = tf.reshape(Y_true, (Y_true.shape + [1]))
        # Y_ = tf.math.log(Y_ / (1. - Y_))
        cost = tf.reduce_mean(tf.square(Y_pred - Y_))
        return cost


class PhysicalModelv2:
    """
    Physical model based one sided Fourier Transform and closure phase operator computed from kpi.BLM (xara)
    Example of use:
        kpi = ExoRIM.Baseline(mask_coordinates) # (x, y) in meters
        cpo = ExoRIM.closure_phase_operator(kpi)
        dftm = ExoRIM.definitions.one_sided_DFTM(kpi.UVC, wavelength, pixels, plate_scale, inv=False, dprec=True):
        phys = PhysicalModelv2(pixels, cpo, dftm)
    """
    def __init__(self, pixels, phase_closure_operator, one_sided_DFTM, dftm_i, SNR):
        """

        :param pixels: Number of pixels on the side of the reconstructed image
        :param bispectrum_projector: Projection operator applied on the visibility to get bispectrum
        :param one_sided_DFTM: Fourier transform operator to get visibility in terms of unique baselines
        :param SNR: Signal to noise ratio for a measurement
        """
        self.pixels = pixels
        self.CPO = tf.constant(phase_closure_operator, dtype=dtype)
        self.A = tf.constant(one_sided_DFTM, dtype=mycomplex)
        self.A_inverse = tf.constant(dftm_i, dtype=mycomplex)
        self.p = one_sided_DFTM.shape[0]  # number of visibility samples
        self.q = phase_closure_operator.shape[0]  # number of independent closure phases
        self.SNR = SNR
        # create matrices that project visibilities to bispectra (V1 = V_{ij}, V_2 = V_{jk} and V_3 = V_{ki})
        bisp_i = np.where(phase_closure_operator != 0)
        V1_i = (bisp_i[0][0::3], bisp_i[1][0::3])
        V2_i = (bisp_i[0][1::3], bisp_i[1][1::3])
        V3_i = (bisp_i[0][2::3], bisp_i[1][2::3])
        self.V1_projector = np.zeros(shape=(self.q, self.p))
        self.V1_projector[V1_i] += 1.0
        self.V1_projector = tf.constant(self.V1_projector, dtype=mycomplex)
        self.V2_projector = np.zeros(shape=(self.q, self.p))
        self.V2_projector[V2_i] += 1.0
        self.V2_projector = tf.constant(self.V2_projector, dtype=mycomplex)
        self.V3_projector = np.zeros(shape=(self.q, self.p))
        self.V3_projector[V3_i] += 1.0
        self.V3_projector = tf.constant(self.V3_projector, dtype=mycomplex)
        self.A1 = tf.tensordot(self.V1_projector, self.A, axes=1)
        self.A2 = tf.tensordot(self.V2_projector, self.A, axes=1)
        self.A3 = tf.tensordot(self.V3_projector, self.A, axes=1)

    def forward(self, image):
        """

        :param image: Tensor of shape (Batch size, pixel, pixel, channels) where channels = 1 for now
        :param flux: Flux vector of size (Batch size)
        :return: A concatenation of complex visibilities and bispectra (dtype: tf.complex128)
        """
        visibilities = self.fourier_transform(image)
        bispectra = self.bispectrum(visibilities)
        y = tf.concat([visibilities, bispectra], axis=1)  # p + q length vectors of type complex128
        return y

    def fourier_transform(self, image):
        im = tf.cast(image, mycomplex)
        flat = tf.keras.layers.Flatten(data_format="channels_last")(im)
        return tf.einsum("...ij, ...j->...i", self.A, flat)  # tensordot broadcasted on batch_size

    def inverse_fourier_transform(self, X):
        amp = tf.cast(X[..., :self.p], mycomplex)
        flat = tf.einsum("...ij, ...j->...i", self.A_inverse, amp)
        flat = tf.square(tf.math.abs(flat))  # intensity is square of amplitude
        flat = tf.cast(flat, dtype)
        return tf.reshape(flat, [-1, self.pixels, self.pixels, 1])

    def bispectrum(self, V):
        V1 = tf.einsum("ij, ...j -> ...i", self.V1_projector, V)
        V2 = tf.einsum("ij, ...j -> ...i", self.V2_projector, V)
        V3 = tf.einsum("ij, ...j -> ...i", self.V3_projector, V)
        return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other methods

    def log_likelihood_v1(self, Y_pred, X):
        """
        :param Y_pred: reconstructed image
        :param X: interferometric data from measurements
        """
        sigma_amp = tf.math.abs(X[..., :self.p]) / self.SNR
        sigma_cp = tf.cast(tf.math.sqrt(3 / self.SNR ** 2), dtype)
        X_pred = self.forward(Y_pred)
        chi2_amp = tf.reduce_mean(tf.square(tf.math.abs(X_pred[..., :self.p] - X[..., :self.p])/(sigma_amp + 1e-6)), axis=1)
        cp_pred = tf.math.angle(X_pred[..., self.p:])
        cp_true = tf.math.angle(X[..., self.p:])
        chi2_cp = tf.reduce_mean(tf.square((cp_pred - cp_true)/(sigma_cp + 1e-6)), axis=1)
        return chi2_amp + chi2_cp

    def grad_log_likelihood_v2(self, Y_pred, X, alpha_amp=None, alpha_vis=1., alpha_bis=None, alpha_cp=None):
        """
        :param Y_pred: reconstructed image
        :param X: interferometric data from measurements (complex vector from forward method)
        """
        sigma_amp, sigma_bis, sigma_cp = self.get_std(X)
        # grad = alpha_amp * chisqgrad_amp(Y_pred, self.A, tf.math.abs(X[..., :self.p]), sigma_amp, self.pixels)
        grad = alpha_vis * chisqgrad_vis(Y_pred, self.A, X[..., :self.p], sigma_amp, self.pixels)
        if alpha_bis is not None:
            grad = grad + alpha_bis * chisqgrad_bs(Y_pred, self.A1, self.A2, self.A3, X[..., self.p:], sigma_bis, self.pixels)
        if alpha_cp is not None:
            grad = grad + alpha_cp * chisqgrad_cphase(Y_pred, self.A1, self.A2, self.A3, tf.math.angle(X[..., self.p:]), sigma_cp, self.pixels)
        return grad

    def grad_log_likelihood_v1(self, Y_pred, X):
        with tf.GradientTape() as tape:
            tape.watch(Y_pred)
            likelihood = self.log_likelihood_v1(Y_pred, X)
        grad = tape.gradient(likelihood, Y_pred)
        return grad

    def get_std(self, X):
        sigma_vis = tf.math.abs(X[..., :self.p]) / self.SNR  # note that SNR should be large (say >~ 10 for gaussian approximation to hold)
        V1 = tf.einsum("ij, ...j -> ...i", self.V1_projector, X[..., :self.p])
        V2 = tf.einsum("ij, ...j -> ...i", self.V2_projector, X[..., :self.p])
        V3 = tf.einsum("ij, ...j -> ...i", self.V3_projector, X[..., :self.p])
        B_amp = tf.cast(tf.math.abs(V1 * tf.math.conj(V2) * V3), dtype)  # same hack from bispectrum
        sigma_cp = tf.cast(tf.math.sqrt(3 / self.SNR**2), dtype)
        sigma_bis = B_amp * sigma_cp
        return sigma_vis, sigma_bis, sigma_cp

    def simulate_noisy_data(self, images):
        batch = images.shape[0]
        X = self.forward(images)
        sigma_vis, sigma_bis, _ = self.get_std(X)
        # noise is picked from a complex normal distribution
        vis_noise_real = tf.random.normal(shape=[batch, self.p], stddev=sigma_vis / 2, dtype=dtype)
        vis_noise_imag = tf.random.normal(shape=[batch, self.p], stddev=sigma_vis / 2, dtype=dtype)
        bis_noise_real = tf.random.normal(shape=[batch, self.q], stddev=sigma_bis / 2, dtype=dtype)
        bis_noise_imag = tf.random.normal(shape=[batch, self.q], stddev=sigma_bis / 2, dtype=dtype)
        vis_noise = tf.complex(vis_noise_real, vis_noise_imag)
        bis_noise = tf.complex(bis_noise_real, bis_noise_imag)
        noise = tf.concat([vis_noise, bis_noise], axis=1)
        return X + noise



# TODO decide if it is reasonable to implement this way in the future: might be more flexible to experiment
class NoiseModel:
    """

    """
    def __init__(self):
        """"""
        pass


class LogLikelihood:
    """

    """

    def __init__(self, chi_squared, regulariser, noises):
        """

        """
        pass
