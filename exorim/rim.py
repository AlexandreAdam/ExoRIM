from .definitions import DTYPE, LOG10
from .utils import nulltape
from .physical_model import PhysicalModel
import tensorflow as tf


class RIM:
    def __init__(self,
                 model,
                 physical_model: PhysicalModel,
                 time_steps=8,
                 adam=True,
                 beta_1=0.9,
                 beta_2=0.99,
                 epsilon=1e-8,
                 log_floor=1e-6  # sets the dynamic range of the model
                 ):
        self.logim = physical_model.logim
        self.pixels = physical_model.pixels
        self.model = model
        self.physical_model = physical_model
        self.steps = time_steps
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.log_floor = log_floor

        if self.logim:
            self.inverse_link_function = tf.keras.layers.Lambda(lambda x: tf.math.log(x + self.log_floor) / LOG10)
            self.link_function = tf.keras.layers.Lambda(lambda x: 10**x)
        else:
            self.link_function = tf.identity
            self.inverse_link_function = tf.identity

        if self.adam:
            self.grad_update = lambda grad, time_step: self.adam_update(grad, time_step)
        else:
            self.grad_update = lambda grad, time_step: grad

    def adam_update(self, grad, time_step):
        self._grad_mean = self. beta_1 * self._grad_mean + (1 - self.beta_1) * grad
        self._grad_var  = self.beta_2 * self._grad_var + (1 - self.beta_2) * tf.square(grad)
        m_hat = self._grad_mean / (1 - self.beta_1**(time_step + 1)) # Unbias moments
        v_hat = self._grad_var / (1 - self.beta_2**(time_step + 1))
        return m_hat / (tf.sqrt(v_hat) + self.epsilon)

    def initial_states(self, batch_size):
        source = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1])
        states = self.model.init_hidden_states(input_pixels=self.pixels, batch_size=batch_size)
        self._grad_mean = tf.zeros_like(source)
        self._grad_var = tf.zeros_like(source)
        return source, states

    def __call__(self, X, sigma, outer_tape=nulltape):
        return self.call(X, sigma, outer_tape)

    def call(self, X, sigma, outer_tape=nulltape):
        """
        Method used in training to get model predictions.

        :param X: Vector of complex visibilities amplitude and closure phases.
        :return: 5D Tensor of shape (steps, batch_size, pixels, pixels, channels)
        """
        batch_size = X.shape[0]
        xt, ht = self.initial_states(batch_size)

        image_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                grad = self.physical_model.grad_log_likelihood(image=xt, X=X, sigma=sigma)
                grad = self.grad_update(grad, time_step=current_step)
            xt, ht = self.model(xt, ht, grad)
            image_series = image_series.write(index=current_step, value=xt)
        return image_series.stack()
