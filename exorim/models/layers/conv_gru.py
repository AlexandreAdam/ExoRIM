import tensorflow as tf
from exorim.definitions import DTYPE


class ConvGRU(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvGRU, self).__init__(dtype=DTYPE, **kwargs)
        self.update_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='sigmoid',
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.reset_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='sigmoid',
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.candidate_activation_gate = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='tanh',
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def __call__(self, features, ht):
        return self.call(features, ht)

    def call(self, features, ht):
        """
        Compute the new state tensor h_{t+1}.
        """
        stacked_input = tf.concat([features, ht], axis=3)
        z = self.update_gate(stacked_input)
        r = self.reset_gate(stacked_input)
        r_state = tf.multiply(r, ht)
        stacked_r_state = tf.concat([features, r_state], axis=3)
        tilde_h = self.candidate_activation_gate(stacked_r_state)
        new_state = tf.multiply(z, ht) + tf.multiply(1 - z, tilde_h)
        return new_state, new_state  # h_{t+1}