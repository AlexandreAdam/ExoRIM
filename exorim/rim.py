from .definitions import DTYPE
from .utils import nulltape
from .physical_model import PhysicalModel
import tensorflow as tf


class RIM:
    def __init__(self,
                 model,
                 physical_model: PhysicalModel,
                 steps=8,
                 adam=True,
                 beta_1=0.9,
                 beta_2=0.99,
                 epsilon=1e-8,
                 ):
        self.logim = physical_model.logim
        self.pixels = physical_model.pixels
        self.model = model
        self.physical_model = physical_model
        self.steps = steps
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.inverse_link_function = physical_model.image_inverse_link
        self.link_function = physical_model.image_link

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
        states = self.model.init_hidden_states(input_pixels=self.pixels, batch_size=batch_size)
        self._grad_mean = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1])
        self._grad_var = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1])
        return states

    def time_step(self, xt, grad, states):
        delta_xt = tf.concat([xt, grad], axis=3)  # concat along channel dimension
        delta_xt, states = self.model(delta_xt, states)
        xt = xt + delta_xt  # RIM update
        return xt, states

    def __call__(self, X, sigma, outer_tape=nulltape):
        return self.call(X, sigma, outer_tape)

    def call(self, X, sigma, outer_tape=nulltape):
        batch_size = X.shape[0]
        ht = self.initial_states(batch_size)
        # Start from dirty beam
        with outer_tape.stop_recording():
            xt = self.physical_model.inverse(X)  # image space
            xt = self.inverse_link_function(xt)  # prediction space (xi)
        image_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                grad, chi_squared = self.physical_model.grad_chi_squared(xi=xt, X=X, sigma=sigma)
                grad = self.grad_update(grad, time_step=current_step)
            xt, ht = self.time_step(xt, grad, ht)
            image_series = image_series.write(index=current_step, value=xt)
            chi_squared_series = chi_squared_series.write(index=current_step, value=chi_squared)
        return image_series.stack(), chi_squared_series.stack()
