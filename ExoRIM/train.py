from ExoRIM.model import RIM, MSE
from ExoRIM.data_generator import SimpleGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ExoRIM.definitions import lossdir, modeldir
from datetime import datetime
from matplotlib import cm
import os

date_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")


class Training:
    def __init__(self):
        self.rim = RIM(steps=4, pixels=32, state_size=4, state_depth=2, noise_std=0.1, num_cell_features=2)
        self.epochs = 10
        self.phys = self.rim.physical_model
        self.generator = SimpleGenerator(30)
        self.loss = MSE()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # utilities for plotting
        self.train_trace = 3  # save a trace of the loss each x batch
        self.test_trace = 1
        self.train_losses = []
        self.test_losses = []
        self.weight_grads = []

        self.trained = False

    def train_weights(self):
        train_trace_loss = 0  # these are tracers, reset at trace percent of an epoch
        test_trace_loss = 0
        for i in range(self.epochs):
            # training
            for image in self.generator.training_batch():
                image = tf.convert_to_tensor(image, dtype=tf.float32)  # x
                noisy_image = self.phys.simulate_noisy_image(image)  # y
                with tf.GradientTape() as tape:
                    tape.watch(self.rim.variables)
                    output = self.rim(noisy_image)
                    cost_value = self.loss(image, output)
                    train_trace_loss += cost_value
                weight_grads = tape.gradient(cost_value, self.rim.trainable_weights)
                if (self.generator.train_index + 1) % self.train_trace == 0:
                    self.train_losses.append(train_trace_loss)
                    print(f"epoch {i+1}: training loss = {train_trace_loss}")
                    self.weight_grads.append(weight_grads)
                    train_trace_loss = 0  # reset the trace
                # prevent exploding gradient
                clipped_grads = [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]
                self.optimizer.apply_gradients(zip(clipped_grads, self.rim.trainable_weights))

            # test
            for image in self.generator.test_batch():
                image = tf.convert_to_tensor(image, dtype=tf.float32)  # x
                noisy_image = self.phys.simulate_noisy_image(image)  # y
                output = self.rim.call(noisy_image)
                cost_value = self.loss.call(x_true=image, x_preds=output)
                test_trace_loss += cost_value
            if (self.generator.test_index + 1) % self.test_trace == 0:
                self.test_losses.append(test_trace_loss)
                print(f"epoch {i+1}: test loss = {test_trace_loss}")
                test_trace_loss = 0  # reset the trace
        self.rim.save_weights(os.path.join(modeldir, f"rim_{date_time}.h5"))
        self.trained = True

    def loss_curve(self):
        if self.trained is False:
            self.train_weights()
        plt.figure()
        trains = np.linspace(0, self.epochs, len(self.train_losses))
        test = np.linspace(0, self.epochs, len(self.test_losses))
        plt.plot(trains, self.train_losses, "k-", label="Training loss")
        plt.plot(test, self.test_losses, "r-", label="Test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(lossdir, f"loss_curve_{date_time}.png"))

    def weights_plot(self):
        if self.trained is False:
            self.train_weights()

        epochs = np.linspace(0, self.epochs, len(self.train_losses))
        mean_weights = [tf.reduce_mean()]
        plt.figure()
        plt.title("Mean gradient per layer")
        plt.style.use('classic')
        fig, axs = plt.subplots()


        # # Create a continuous norm to map from data points to colors
        # norm = plt.Normalize(u[:, 0].min(), u[:, 0].max())
        # lc = LineCollection(segments, cmap='autumn', norm=norm)
        # lc.set_array(u[:, 0])
        # lc.set_linewidth(3)
        # line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)


if __name__ == "__main__":
    train = Training()
    train.loss_curve()
