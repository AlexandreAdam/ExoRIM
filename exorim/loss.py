import tensorflow as tf


class Loss(tf.keras.losses.Loss):
    def __init__(self, logim, tv_beta=0., *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)
        self.logim = logim
        self.mse = lambda Y_pred, Y_true: tf.reduce_mean(tf.square(tf.expand_dims(Y_true, 0) - Y_pred), axis=(0, 2, 3, 4))
        self.tv = lambda Y_pred: tf.image.total_variation(Y_pred[-1, ...])
        self.tv_beta = tv_beta

    def __call__(self, Y_pred, Y_true, *args, **kwargs):
        return self.call(Y_pred, Y_true)

    def call(self, Y_pred, Y_true):
        cost = self.mse(Y_pred, Y_true)
        if self.tv_beta:
            if self.logim:
                Y_pred = tf.math.exp(Y_pred)
            cost = cost + self.tv_beta * self.tv(Y_pred)
        return tf.reduce_sum(cost)


class MSE(tf.keras.losses.Loss):
    def __init__(self):
        super(MSE, self).__init__()

    def __call__(self, Y_pred, Y_true, *args, **kwargs):
        return self.call(Y_pred, Y_true)

    def call(self, Y_pred, Y_true):
        """
        :param Y_true: 4D tensor to be compared with Y_preds
        :param Y_preds: 5D tensor output of the call method
        :return: Score
        """
        cost = tf.reduce_mean(tf.square(Y_pred - tf.expand_dims(Y_true, 0) ))
        return cost
