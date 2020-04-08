from ExoRIM.dataset import CenteredDataset
from ExoRIM.model import RIM, MSE
from ExoRIM.definitions import modeldir, dtype
#from ExoRIM.training_visualization import TrainViz
#from ExoRIM._train_master import TrainMaster
import tensorflow as tf
import numpy as np
import os



    def _train_batch(self, image, loss_trace):

        noisy_image = self.phys.simulate_noisy_image(image)  # y
        with tf.GradientTape() as tape:
            tape.watch(self.rim.trainable_weights)
            output = self.rim(noisy_image)
            cost_value = self.loss(image, output)
            cost_value += tf.reduce_sum(tf.square(self.rim.losses))  # add L2 regularisation to loss
            loss_trace.append(cost_value)
        weight_grads = tape.gradient(cost_value, self.rim.trainable_weights)
        # prevent exploding gradient
        clipped_grads = tf.clip_by_global_norm(weight_grads, 10)  # [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]
        #self.weight_grads.append(clipped_grads)
        self.optimizer.apply_gradients(zip(clipped_grads, self.rim.trainable_weights))
        return output, loss_trace

    def _test_batch(self, image, loss_trace):
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # x
        noisy_image = self.phys.simulate_noisy_image(image)  # y
        output = self.rim.call(noisy_image)
        cost_value = self.loss.call(x_true=image, x_preds=output)
        cost_value += tf.reduce_sum(tf.square(self.rim.losses))  # add L2 regularisation to loss
        loss_trace.append(cost_value)
        return output, loss_trace

    def _epoch(self, epoch, verbose=1):
        train_loss_trace = []  # these are tracers to compute loss, reset each self.train_trace batch
        test_loss_trace = []
        # training
        for image in self.generator.training_batch():
            output, train_trace_loss = self._train_batch(image, train_loss_trace)
            if self.generator.train_index in self.train_traces:
                self.train_losses.append(np.mean(train_trace_loss)/self.generator.train_batch_size)
                self.snap(output, image, state="train")
                if verbose > 0:
                    print(f"epoch {epoch + 1}: training loss = {np.mean(train_trace_loss)/self.generator.train_batch_size}")
                train_loss_trace = []  # reset the trace
        # test
        for image in self.generator.test_batch():
            output, test_loss_trace = self._test_batch(image, test_loss_trace)
            if self.generator.test_index in self.test_traces:
                self.test_losses.append(np.mean(test_loss_trace)/self.generator.test_batch_size)  # we plot the mean of the loss
                self.snap(output, image, state="test")
                if verbose > 0:
                    print(f"epoch {epoch + 1}: test loss = {np.mean(test_loss_trace)/self.generator.test_batch_size}")
                test_loss_trace = []  # reset the trace

    def train_weights(self):
        for epoch in range(self.epochs):
            self._epoch(epoch)
        self.rim.save_weights(os.path.join(modeldir, f"{self.model_name}_{self.init_time}.h5"))
        self.trained = True
        # self.trainable = False #- eventually to freeze the BatchNorm in place


if __name__ == "__main__":
    train = Training()
    train.train_weights()
    train.save_movies("RIM")
    train.loss_curve()