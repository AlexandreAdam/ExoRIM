import tensorflow as tf
import numpy as np
from .definitions import kernal_reg_amp, bias_reg_amp, kernel_size, dtype, initializer
from .pysco.kpi import kpi


class ConvGRU(tf.keras.Model):
    def __init__(self, num_features, kernel_size=kernel_size):
        super(ConvGRU, self).__init__()
        num_filters = num_features
        self.update_gate = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[kernel_size, kernel_size],
            strides=1,
            activation='sigmoid',
            padding='same'
        )
        self.reset_gate = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[kernel_size, kernel_size],
            strides=1,
            activation='sigmoid',
            padding='same'
        )
        self.candidate_activation_gate = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[kernel_size, kernel_size],
            strides=1,
            activation='tanh',
            padding='same'
        )

    def call(self, features, ht):
        """
        Compute the new state tensor.
        :param inputs: List = [features(xt), ht]
        :return:
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
    def __init__(self, state_size, num_cell_features, dtype=dtype):
        super(Model, self).__init__(dtype=dtype)
        self._state_size = state_size
        self.num_gru_features = num_cell_features // 2  # This is because we split the state tensor in 2
        num_filt_emb1_1 = self.num_gru_features
        num_filt_emb1_2 = self.num_gru_features
        num_filt_emb3_1 = self.num_gru_features
        num_filt_emb3_2 = self.num_gru_features
        strides_conv1 = 2 # Make this not as arbitrary, function of pixels
        strides_conv2 = 2
        strides_conv3 = 1

        self.conv1_1 = tf.keras.layers.Conv2D(
            filters=num_filt_emb1_1,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv1,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer,
            data_format="channels_last"
        )
        self.conv1_2 = tf.keras.layers.Conv2D(
            filters=num_filt_emb1_2,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv2,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer,
            data_format="channels_last"
        )
        self.conv1_3 = tf.keras.layers.Conv2D(
            filters=num_filt_emb1_2,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv3,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer,
            data_format="channels_last"
        )
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=self.num_gru_features,
            kernel_size=[kernel_size, kernel_size],
            strides=1,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer
        )
        self.conv3_1 = tf.keras.layers.Conv2DTranspose(
            filters=num_filt_emb3_1,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv1,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer
        )
        self.conv3_2 = tf.keras.layers.Conv2DTranspose(
            filters=num_filt_emb3_2,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv2,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer
        )
        self.conv3_3 = tf.keras.layers.Conv2DTranspose(
            filters=num_filt_emb3_2,
            kernel_size=[kernel_size, kernel_size],
            strides=strides_conv3,
            activation=tf.keras.layers.LeakyReLU(),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=[kernel_size, kernel_size],
            strides=1,
            activation='linear',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l=kernal_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
            kernel_initializer=initializer
        )
        self.gru1 = ConvGRU(self.num_gru_features)
        self.gru2 = ConvGRU(self.num_gru_features)

    def call(self, xt, ht, grad):
        """
        :param inputs: List = [xt, ht, grad]
            xt: Image tensor of shape (batch size, num of pixel, num of pixel, filters=1)
            ht: Hidden state list: [h^0_t, h^1_t, ...]
                    (batch size, downsampled image size, downsampled image size, state_size)
                The downsampled image size is defined num_cell_features in RIM
            grad: Gradient of the log-likelihood function for y (data) given xt
        :return: x_{t+1}, h_{t+1}
        """
        stacked_input = tf.concat([xt, grad], axis=3)
        features = self.conv1_1(stacked_input)
        features = self.conv1_2(features)
        features = self.conv1_3(features)  # shape needs to match ht_11 and ht_12, make sure strides add up
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(features, ht_1)  # to be recombined in new state
        ht_1_features = self.conv2(ht_1)
        ht_2 = self.gru2(ht_1_features, ht_2)
        delta_xt = self.conv3_1(ht_2)  # conv3 is transposed convolution, brings back the tensor to image shape
        delta_xt = self.conv3_2(delta_xt)
        delta_xt = self.conv3_3(delta_xt)
        delta_xt = self.conv4(delta_xt)
        xt_1 = xt + delta_xt
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return xt_1, new_state


class RIM(tf.keras.Model):
    def __init__(self, steps, pixels, noise_std, state_size,
                 state_depth, num_cell_features, channels=1, dtype=dtype):
        super(RIM, self).__init__(dtype=dtype)
        assert state_size % 2 == 0, "State size has to be a multiple of 2"
        self._dtype = dtype
        self.channels = channels
        self.pixels = pixels
        self.steps = steps
        self.state_size = state_size
        self.state_depth = state_depth
        self.model = Model(state_size=state_size, num_cell_features=num_cell_features, dtype=self._dtype)
        self.gradient_instance_norm = tf.keras.layers.BatchNormalization(axis=1)  # channel must be last dimension of input for this layer
        self.physical_model = PhysicalModel(pixels=pixels, noise_std=noise_std)
        self.noise_std = noise_std
        self.trainable = True

    def call(self, y):
        """

        :param y: Vector of complex visibilities amplitude and closure phases
        :return: 4D Tensor of shape (batch_size, [image_size], steps)
        """
        batch_size = y.shape[0]
        x0 = self.initial_guess(batch_size)
        h0 = self.init_hidden_states(batch_size)
        # Compute the gradient through auto-diff since model is highly non-linear
        with tf.GradientTape() as g:
            g.watch(x0)
            likelihood = self.log_likelihood(x0, y)
        grad = self.gradient_instance_norm(g.gradient(likelihood, x0))
        #grad = g.gradient(likelihood, x0)
        xt, ht = self.model(x0, h0, grad)
        outputs = tf.reshape(xt, xt.shape + [1])  # Plus one dimension for step stack
        for current_step in range(self.steps - 1):
            with tf.GradientTape() as g:
                g.watch(xt)
                likelihood = self.log_likelihood(xt, y)
            grad = self.gradient_instance_norm(g.gradient(likelihood, xt))
            #grad = g.gradient(likelihood, xt)
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

        #TODO We should think to add a small paramater epsilon to the division in case variance get too small
        :param y: True vector of complex visibility amplitudes and closure phases
        :param xt: Image reconstructed at step t of the reconstruction
        :return: Scalar L
        """
        yhat = self.physical_model.physical_model(xt)
        return 0.5 * tf.math.reduce_sum(tf.square(y - yhat)) / self.noise_std**2  #* tf.ones_like(xt)


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

    def call(self, x_true, x_preds):
        """
        Dimensions are
        0: batch_size
        1: x dimension of image
        2: y dimension of image
        3: channels (=1 for gray image)
        4: time steps of the RIM output
        :param x_true:
        :param x_preds:
        :return:
        """
        x_true_ = tf.reshape(x_true, (x_true.shape + [1]))
        cost = tf.reduce_sum(tf.square(x_true_ - x_preds), axis=[1, 2, 3])
        cost = tf.reduce_sum(cost, axis=1)  # Sum over time steps (this one could be weighted)
        cost = tf.reduce_sum(cost, axis=0)  # Sum over the batch
        return cost


class PhysicalModel(object):

    def __init__(self, pixels, noise_std):
        """
        :param pixels: Number of pixel on the side of a square camera
        :param noise_std: Standard deviation of the noise
        """
        print('initializing Phys_Mod')
        self.pixels = pixels
        self.noise_std = noise_std
        try:
            bs = kpi(file='coords.txt', bsp_mat='sparse')
            print('Loaded coords.txt')
        except:
            coords = np.random.randn(6, 2) # TODO
            #plt.plot(coords[:, 0], coords[:, 1], '.')
            np.savetxt('coords.txt', coords)
            print('Generated random array and saved to coords.txt')
            bs = kpi(file='coords.txt', bsp_mat='sparse')
            print('Loaded coords.txt')

        ## create p2vm matrix

        x = np.arange(self.pixels)
        xx, yy = np.meshgrid(x, x)

        p2vm_sin = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        for j in range(bs.uv.shape[0]):
            p2vm_sin[j, :] = np.ravel(np.sin(xx * bs.uv[j, 0] + yy * bs.uv[j, 1]))

        p2vm_cos = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        for j in range(bs.uv.shape[0]):
            p2vm_cos[j, :] = np.ravel(np.cos(xx * bs.uv[j, 0] + yy * bs.uv[j, 1]))

        # create tensor to hold cosine and sine projection operators
        self.cos_projector = tf.constant(p2vm_cos.T, dtype=dtype)
        self.sin_projector = tf.constant(p2vm_sin.T, dtype=dtype)
        self.bispectra_projector = tf.constant(bs.uv_to_bsp, dtype=dtype)

        vis2s = np.zeros(p2vm_cos.shape[0])
        closure_phases = np.zeros(bs.uv_to_bsp.shape[0])
        self.vis2s_tensor = tf.constant(vis2s, dtype=dtype)
        self.cp_tensor = tf.constant(closure_phases, dtype=dtype)
        self.data_tensor = tf.concat([self.vis2s_tensor, self.cp_tensor], 0)

        # create tensor to hold your uncertainties
        #self.vis2s_err_tensor = tf.constant(np.ones_like(vis2s), dtype=dtype)
        self.vis2s_error = tf.random.normal(stddev=self.noise_std, shape=tf.shape(vis2s), dtype=dtype)
        self.cp_err_tensor = tf.constant(np.ones_like(closure_phases), dtype=dtype)
        #self.error_tensor = tf.concat([self.vis2s_err_tensor, self.cp_err_tensor], axis=0)

    def physical_model(self, image):
        if not tf.is_tensor(image):
            tfim = tf.constant(image, dtype=dtype)
        else:
            tfim = image
        flat = tf.keras.layers.Flatten(data_format="channels_last")(tfim)
        sin_model = tf.tensordot(flat, self.sin_projector, axes=1)
        cos_model = tf.tensordot(flat, self.cos_projector, axes=1)
        visibility_squared = tf.square(sin_model) + tf.square(cos_model)
        #phases = tf.math.angle(tf.complex(cos_model, sin_model))
        #closure_phase = tf.tensordot(self.bispectra_projector, phases, axes=1)
        #y = tf.concat([visibility_squared, closure_phase], axis=0)
        y = visibility_squared
        return y

    def simulate_noisy_image(self, image):
        out = self.physical_model(image)
        out += self.vis2s_error
        out = (out - tf.math.reduce_min(out)) / (tf.math.reduce_max(out) - tf.math.reduce_min(out)) # normalize
        return out
    
