from ExoRIM.dataset import CenteredDataset
from ExoRIM.model import RIM, MSE
from ExoRIM.definitions import modeldir
import tensorflow as tf
from datetime import datetime
import os
import warnings
from celluloid import Camera
import matplotlib.gridspec as gridspec
from ExoRIM.definitions import image_dir
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.rcParams["figure.figsize"] = (10, 10)


class Training:
    def __init__(
            self,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=MSE(),
            model_name="RIM",
            epochs=500,
            total_items=1000,
            split=0.8,
            batch_size=20,
            checkpoints=None,
            images_saved=10,
            steps=12,  # number of steps for the reconstruction
            pixels=32,
            state_size=8,  # hidden state 2D size
            state_depth=2,  # Channel dimension of hidden state
            noise_std=0.0001, # This is relative to the smallest complex visibility
            num_cell_features=2,
            step_trace=[3, 8, 11], # starting from 0
            number_of_images=1
    ):
        self.init_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.model_name = model_name
        self.model_args = {
            "steps": steps,
            "pixels": pixels,
            "state_size": state_size,
            "state_depth": state_depth,
            "noise_std": noise_std,
            "num_cell_features": num_cell_features
        }
        self.rim = RIM(**self.model_args)
        self.epochs = epochs
        self.steps = steps
        self.phys = self.rim.physical_model
        self.dataset = CenteredDataset(
            self.phys,
            total_items=total_items,
            split=split,
            train_batch_size=batch_size,
            pixels=pixels
        )
        self.loss = loss
        self.optimizer = optimizer

        # utilities for plotting
        self.train_losses = tf.zeros(epochs)
        self.test_losses = tf.zeros(epochs)
        self.step_trace = step_trace
        self.images = np.zeros((epochs, pixels, pixels, len(step_trace)))
        self.true_image = np.zeros((pixels, pixels))
        self.loss_trace = tf.keras.metrics.Mean()
        self.loss_traces = np.zeros(epochs)

    def _epoch(self, epoch, verbose=1):
        # training
        for noisy_image in self.dataset.noisy_train_set:
            true_image = next(iter(self.dataset.true_train_set))
            self.loss_trace.reset_states()
            with tf.GradientTape() as tape:
                tape.watch(self.rim.trainable_weights)
                output = self.rim(noisy_image)
                cost_value = self.loss(true_image, output)
                cost_value += tf.reduce_sum(tf.square(self.rim.losses))  # add L2 regularisation to loss
                self.loss_trace.update_state(cost_value)
            weight_grads = tape.gradient(cost_value, self.rim.trainable_weights)
            # prevent exploding gradient
            clipped_grads, _ = tf.clip_by_global_norm(weight_grads, 10)  # [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]
            self.optimizer.apply_gradients(zip(clipped_grads, self.rim.trainable_weights))
        self.loss_traces[epoch] += self.loss_trace.result()
        for i, trace in enumerate(self.step_trace):
            self.images[epoch, :, :, i] += output[0, :, :, 0, trace]
        if epoch == 0:
            self.true_image += true_image[0, :, :, 0]

    def train_weights(self):
        for epoch in range(self.epochs):
            self._epoch(epoch)
        self.rim.save_weights(os.path.join(modeldir, f"{self.model_name}_{self.init_time}.h5"))
        self.trained = True
        # self.trainable = False #- eventually to freeze the BatchNorm in place

    def save_movies(self, title):
        # Utilities to film output during training
        self.movie_fig = plt.figure()
        self.cam = Camera(self.movie_fig)

        widths = [10] * (len(self.step_trace) + 1) + [1]
        self.gs = gridspec.GridSpec(2, len(self.step_trace) + 2,  width_ratios=widths)
        self.loss_x_axis = list(range(1, self.epochs+1))

        for i in range(self.epochs):
            loss_ax = self.movie_fig.add_subplot(self.gs[-1, :])
            cbar_ax = self.movie_fig.add_subplot(self.gs[:-1, -1])
            self.movie_fig.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.get_cmap("gray")),
                cax=cbar_ax
            )
            loss_ax.set_xlabel("Epochs")
            loss_ax.set_ylabel("Average Loss (over all samples)")

            loss_ax.plot(self.loss_x_axis[:i+1], self.loss_traces[:i+1], "k-")

            # plot the ground truth to the right
            g_truth = self.movie_fig.add_subplot(self.gs[0, -2])
            g_truth.get_xaxis().set_visible(False)
            g_truth.get_yaxis().set_visible(False)
            g_truth.imshow(self.true_image, cmap="gray")
            g_truth.set_title("Ground Truth")
            # plot the images at different reconstruction stages according to step_traces
            for col, step in enumerate(self.step_trace):
                ax = self.movie_fig.add_subplot(self.gs[0, col])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.imshow(self.images[i, :, :, col], cmap="gray")
                ax.set_title(f"Step {step}")
            # Smile!
            self.cam.snap()
        self.cam.animate().save(os.path.join(image_dir, f"{title}_train_{self.init_time}.mp4"), writer="ffmpeg")

if __name__ == "__main__":
    coords = np.random.randn(20, 2)
    np.savetxt("coords.txt", coords)
    train = Training()
    train.train_weights()
    train.save_movies("RIM")
