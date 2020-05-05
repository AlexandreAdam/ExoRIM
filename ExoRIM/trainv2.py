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
from matplotlib.lines import Line2D
import pickle
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.rcParams["figure.figsize"] = (10, 10)


class Training:
    def __init__(
            self,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=MSE(),
            model_name="RIM",
            epochs=20,
            total_items=30,
            split=0.8,
            batch_size=2,
            checkpoints=None,
            images_saved=1,
            steps=12,  # number of steps for the reconstruction
            pixels=32,
            state_size=8,  # hidden state 2D size
            state_depth=2,  # Channel dimension of hidden state
            noise_std=0.0001, # This is relative to the smallest complex visibility
            num_cell_features=2,
            step_trace=[3, 8, 11]  # index of step
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
        self.step_trace = step_trace
        self.images = np.zeros((epochs, images_saved, len(step_trace), 2, pixels, pixels))
        self._image_saved = images_saved
        self.true_images = np.zeros((images_saved, pixels, pixels))
        self.loss_trace = tf.keras.metrics.Mean()
        self.losses = np.zeros((2, epochs))

    def _train_batch(self, epoch, batch, noisy_image, true_image):
        with tf.GradientTape() as tape:
            tape.watch(self.rim.trainable_weights)
            output = self.rim(noisy_image)
            cost_value = self.loss(true_image, output)
            cost_value += tf.reduce_sum(tf.square(self.rim.losses))  # add L2 regularisation to loss
            self.loss_trace.update_state(cost_value / noisy_image.shape[0]) # cost per sample
        weight_grads = tape.gradient(cost_value, self.rim.trainable_weights)
        # prevent exploding gradient
        clipped_grads, _ = tf.clip_by_global_norm(weight_grads, 10)
        self.optimizer.apply_gradients(zip(clipped_grads, self.rim.trainable_weights))

        if batch == 0:
            for step, trace in enumerate(self.step_trace):
                self.images[epoch, :, step, 0, :, :] += output[:self._image_saved, :, :, 0, trace]  # [batch, pix, pix, channel, steps]

    def _epoch(self, epoch):
        # training
        self.loss_trace.reset_states()
        for batch, noisy_image in enumerate(self.dataset.noisy_train_set):
            true_image = next(iter(self.dataset.true_train_set))
            self._train_batch(epoch, batch, noisy_image, true_image)
        self.losses[0, epoch] += self.loss_trace.result()
        output = self.rim(self.dataset.noisy_test_set)
        cost_value = self.loss(self.dataset.true_test_set, output)
        cost_value += tf.reduce_sum(tf.square(self.rim.losses))
        self.losses[1, epoch] += cost_value / self.dataset.noisy_test_set.shape[0]
        if epoch == 0:
            self.true_images += true_image[:self._image_saved, :, :, 0].numpy()
        for step, trace in enumerate(self.step_trace):
            self.images[epoch, :, step, 1, :, :] += output[:self._image_saved, :, :, 0, trace]

    def train_weights(self):
        for epoch in range(self.epochs):
            self._epoch(epoch)
        self.trained = True
        # self.trainable = False #- eventually to freeze the BatchNorm in place

    def save_movies(self, title):
        # Utilities to film output during training
        run_dir = os.path.join(image_dir, f"{self.model_name}_{self.init_time}")
        os.mkdir(run_dir)
        for image in range(self._image_saved):
            labels = ['Single SGD', 'SGD with momentum', 'SGD with Nesterov',
                      'SGD with Adagrad', 'SGD with Adadelta', 'SGD with RMSprob', 'SGD with Adam',
                      'SGD with Adamax', 'SGD with Nadam', 'SGD with AMSgrad']
            colors = ['k', 'r']
            handles = []
            for c, l in zip(colors, labels):
                handles.append(Line2D([0], [0], color=c, label=l))

            plt.legend(handles=handles, loc='upper left')
            for state in range(2):# test and train
                movie_fig = plt.figure()
                cam = Camera(movie_fig)
                movie_fig.suptitle('This is a somewhat long figure title', fontsize=16)

                widths = [10] * (len(self.step_trace) + 1) + [1]
                gs = gridspec.GridSpec(2, len(self.step_trace) + 2,  width_ratios=widths)
                loss_x_axis = list(range(1, self.epochs+1))

                for epoch in range(self.epochs):
                    movie_fig.suptitle(title, fontsize=16)
                    loss_ax = movie_fig.add_subplot(gs[-1, :])
                    cbar_ax = movie_fig.add_subplot(gs[:-1, -1])
                    movie_fig.colorbar(
                        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.get_cmap("gray")),
                        cax=cbar_ax
                    )
                    loss_ax.set_xlabel("Epochs")
                    loss_ax.set_ylabel("Average Loss")

                    loss_ax.plot(loss_x_axis[:epoch + 1], self.losses[0, :epoch + 1], "k-", label="Training set")
                    loss_ax.plot(loss_x_axis[:epoch + 1], self.losses[1, :epoch + 1], "r-", label="Test set")
                    if epoch == 0:
                        loss_ax.legend()


                    # plot the ground truth to the right
                    g_truth = movie_fig.add_subplot(gs[0, -2])
                    g_truth.get_xaxis().set_visible(False)
                    g_truth.get_yaxis().set_visible(False)
                    g_truth.imshow(self.true_images[image, :, :], cmap="gray")
                    g_truth.set_title("Ground Truth")
                    # plot the images at different reconstruction stages according to step_traces
                    for col, step in enumerate(self.step_trace):
                        ax = movie_fig.add_subplot(gs[0, col])
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.imshow(self.images[epoch, image, col, state, :, :], cmap="gray")
                        ax.set_title(f"Step {step+1}")
                    # Smile!
                    cam.snap()
                if state == 0: _state = "train"
                else: _state = "test"
                cam.animate().save(os.path.join(run_dir, f"({image+1})_{title}_{_state}_{self.init_time}.mp4"), writer="ffmpeg")

        with open(os.path.join(run_dir, f"{title}_{self.init_time}.pickle"), 'wb') as f:
            data = {"images": self.images, "loss": self.losses}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    coords = np.random.randn(10, 2)
    np.savetxt("coords.txt", coords)
    train = Training()
    train.train_weights()
    train.rim.save_weights(os.path.join(modeldir, f"{train.model_name}_{train.init_time}.h5"))
    train.save_movies("RIM")