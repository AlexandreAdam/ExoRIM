import tensorflow as tf


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


class KLDivergence(tf.keras.losses.Loss):
    def __init__(self, floor=1e-2):
        super(KLDivergence, self).__init__()
        self.floor = floor
        # Basically the same has keras KL divergence loss with explicit numerical floor to prevent
        # division by zero and log(0)

    def call(self, Y_pred, Y_true):
        """
        :param Y_true: 4D tensor to be compared with Y_preds
        :param Y_preds: 5D tensor output of the RIM call method
        :return: Score
        """
        Y_ = tf.reshape(Y_true, (Y_true.shape + [1])) + self.floor
        cost = tf.reduce_sum(Y_ * tf.math.log(Y_ / (Y_pred + self.floor)), axis=[1, 2, 3])
        cost = tf.reduce_mean(cost, axis=1)  # over reconstruction time steps
        cost = tf.reduce_mean(cost)          # over batch
        return cost


class MAE(tf.keras.losses.Loss):
    def __init__(self):
        super(MAE, self).__init__()

    def call(self, Y_pred, Y_true):
        """
        :param Y_true: 4D tensor to be compared with Y_preds
        :param Y_preds: 5D tensor output of the call method
        :return: Score
        """
        Y_ = tf.reshape(Y_true, (Y_true.shape + [1]))
        # Y_ = tf.math.log(Y_ / (1. - Y_))
        cost = tf.reduce_mean(tf.abs(Y_pred - Y_))
        return cost

    def test(self, Y_pred, Y_true):
        """
        :param Y_true: 4D tensor to be compared with Y_preds
        :param Y_preds: 4D tensor output (last step of the reconstruction for test)
        :return: Score: A vector of the same dimension as Y.shape[0]
        """
        cost = tf.reduce_mean(tf.abs(Y_pred - Y_true))
        return cost
