import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
from operator import add
from functools import reduce
from celluloid import Camera
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.rcParams["figure.figsize"] = (10, 10)


class TrainViz:
    """
    Tools to plot loss curve and movie of the reconstruction through the training of the model.
    """
    def __init__(self, step_trace: list, number_of_images=1, **kwargs):
        super(TrainViz, self).__init__(**kwargs)
        self.image_dir = os.path.join(image_dir, f"{self.model_name}_{self.init_time}")
        os.mkdir(self.image_dir)

        # Utility to select a number of images in a batch to plot in a single figure
        self._train_images, self._test_images = self._image_in_batch(number_of_images)

        # Utilities to film output during training
        self.movie_figs = {
            "train": dict(zip(self.train_traces, map(plt.figure, self.train_traces))),
            "test": dict(zip(self.test_traces, map(plt.figure, self.test_traces)))
        }
        self.cams = {
            "train": dict(zip(self.train_traces, map(Camera, self.movie_figs["train"].values()))),
            "test": dict(zip(self.test_traces, map(Camera, self.movie_figs["test"].values())))
        }

        widths = [10] * (len(step_trace) + 1) + [1]
        self.gs_train = gridspec.GridSpec(len(self._train_images) + 1, len(step_trace) + 2,  width_ratios=widths)
        self.gs_test = gridspec.GridSpec(len(self._test_images) + 1, len(step_trace) + 2,  width_ratios=widths)
        self.step_trace = step_trace
        self.loss_x_axis = self._loss_x_axis()

    def _loss_x_axis(self):
        # compute the percentage of an epoch that each trace represent, in term of number of batches
        traces = np.array([batch/self.generator.train_batches_in_epoch for batch in np.sort(self.train_traces)])
        # broadcast the percentages to each epochs, then flatten array
        train_x_axis = reduce(add, [(epoch + traces).tolist() for epoch in range(0, self.epochs)])
        traces = np.array([batch/self.generator.test_batches_in_epoch for batch in np.sort(self.test_traces)])
        test_x_axis = reduce(add, [(epoch + traces).tolist() for epoch in range(0, self.epochs)])
        return {"train": train_x_axis, "test": test_x_axis}

    def _image_in_batch(self, number_of_images):
        """
        Choose the images in the batch to be saved, according to number_of_images and the batch_size. This
        function make sure we select the number of images selected by the user in a single figure, while
        taking into account the number of images that is actually in a batch.
        :return: index list
        """
        np.random.seed(42)
        train_batch_size = self.generator.train_batch_size
        test_batch_size = self.generator.test_batch_size
        train_pool = list(range(train_batch_size))
        test_pool = list(range(test_batch_size))
        train_num = min(train_batch_size, number_of_images)
        test_num = min(test_batch_size, number_of_images)
        train_images = np.random.choice(train_pool, train_num, replace=False)
        test_images = np.random.choice(test_pool, test_num, replace=False)
        return train_images, test_images

    def snap(self, output, true_image, state: str):
        """
        Place the output in the correct axe and the correct figure
        :param output: Output tensor of the model, of shape (batch_size, pixels, pixels, channels, steps)
        :param true_image: True image to be reconstructed, of shape (batch_size, pixels, pixels, channels)
        :param train: Switch, determine whether we are in train phase or test phase
        :param state: Either "train" or "test"
        """

        if state == "train":
            fig = self.generator.train_index  # correspond to a train_trace
            gs = self.gs_train
            images = self._train_images
            losses = self.train_losses
        else:
            gs = self.gs_test
            fig = self.generator.test_index
            images = self._test_images
            losses = self.test_losses
        loss_ax = self.movie_figs[state][fig].add_subplot(gs[-1, :])
        cbar_ax = self.movie_figs[state][fig].add_subplot(gs[:-1, -1])
        self.movie_figs[state][fig].colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.get_cmap("gray")),
            cax=cbar_ax
        )
        loss_ax.set_xlabel("Epochs")
        loss_ax.set_ylabel("Average Loss (over all samples)")

        loss_ax.plot(self.loss_x_axis[state][:len(losses)], losses, "k-")

        for row, image in enumerate(images):
            # plot the ground truth to the right
            g_truth = self.movie_figs[state][fig].add_subplot(gs[row, -2])
            g_truth.get_xaxis().set_visible(False)
            g_truth.get_yaxis().set_visible(False)
            g_truth.imshow(true_image[image, :, :, 0], cmap="gray")
            if row == 0:
                g_truth.set_title("Ground Truth")
            # plot the images at different reconstruction stages according to step_traces
            for col, step in enumerate(self.step_trace):
                ax = self.movie_figs[state][fig].add_subplot(gs[row, col])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.imshow(output[image, :, :, 0, step], cmap="gray")
                if row == 0:
                    ax.set_title(f"Step {step}")
        # Smile!
        self.cams[state][fig].snap()

    def save_movies(self, title: str):
        # ffmpeg -i input.mov -r 0.25 output_%04d.png -- terminal command to unpack movie into jpg
        for i, camera in enumerate(self.cams["train"].values()):
            camera.animate().save(os.path.join(self.image_dir, f"{title}_train{i+1}.mp4"), writer="ffmpeg")
        for i, camera in enumerate(self.cams["test"].values()):
            camera.animate().save(os.path.join(self.image_dir, f"{title}_test{i+1}.mp4"), writer="ffmpeg")

    def loss_curve(self):
        plt.figure()
        plt.plot(self.loss_x_axis["train"], self.train_losses, "k-", label="Training loss")
        plt.plot(self.loss_x_axis["test"], self.test_losses, "r-", label="Test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(lossdir, f"loss_curve_{self.init_time}.png"))

    # def weights_plot(self):
    #     if self.trained is False:
    #         self.train_weights()
    #
    #     epochs = np.linspace(0, self.epochs, len(self.train_losses))
    #     mean_weights = [tf.reduce_mean()]
    #     plt.figure()
    #     plt.title("Mean gradient per layer")
    #     plt.style.use('classic')
    #     fig, axs = plt.subplots()

        # # Create a continuous norm to map from data points to colors
        # norm = plt.Normalize(u[:, 0].min(), u[:, 0].max())
        # lc = LineCollection(segments, cmap='autumn', norm=norm)
        # lc.set_array(u[:, 0])
        # lc.set_linewidth(3)
        # line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)


  # def save_movies(self, title):
  #       # Utilities to film output during training
  #       run_dir = os.path.join(image_dir, f"{self.model_name}_{self.init_time}")
  #       os.mkdir(run_dir) for image in range(self._image_saved):
  #           labels = ["Train", "Test"]
  #           colors = ['k', 'r']
  #           handles = []
  #           for c, l in zip(colors, labels):
  #               handles.append(Line2D([0], [0], color=c, label=l))
  #           plt.legend(handles=handles, loc='upper left')
  #           for state in range(2):# test and train
  #               movie_fig = plt.figure()
  #               cam = Camera(movie_fig)
  #               movie_fig.suptitle('This is a somewhat long figure title', fontsize=16)
  #
  #               widths = [10] * (len(self.step_trace) + 1) + [1]
  #               gs = gridspec.GridSpec(2, len(self.step_trace) + 2,  width_ratios=widths)
  #               loss_x_axis = list(range(1, self.epochs+1))
  #
  #               for epoch in range(self.epochs):
  #                   movie_fig.suptitle(title, fontsize=16)
  #                   loss_ax = movie_fig.add_subplot(gs[-1, :])
  #                   cbar_ax = movie_fig.add_subplot(gs[:-1, -1])
  #                   movie_fig.colorbar(
  #                       plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.get_cmap("gray")),
  #                       cax=cbar_ax
  #                   )
  #                   loss_ax.set_xlabel("Epochs")
  #                   loss_ax.set_ylabel("Average Loss")
  #
  #                   loss_ax.plot(loss_x_axis[:epoch + 1], self.losses[0, :epoch + 1], "k-", label="Training set")
  #                   loss_ax.plot(loss_x_axis[:epoch + 1], self.losses[1, :epoch + 1], "r-", label="Test set")
  #                   if epoch == 0:
  #                       loss_ax.legend()
  #
  #
  #                   # plot the ground truth to the right
  #                   g_truth = movie_fig.add_subplot(gs[0, -2])
  #                   g_truth.get_xaxis().set_visible(False)
  #                   g_truth.get_yaxis().set_visible(False)
  #                   g_truth.imshow(self.true_images[image, :, :], cmap="gray")
  #                   g_truth.set_title("Ground Truth")
  #                   # plot the images at different reconstruction stages according to step_traces
  #                   for col, step in enumerate(self.step_trace):
  #                       ax = movie_fig.add_subplot(gs[0, col])
  #                       ax.get_xaxis().set_visible(False)
  #                       ax.get_yaxis().set_visible(False)
  #                       ax.imshow(self.images[epoch, image, col, state, :, :], cmap="gray")
  #                       ax.set_title(f"Step {step+1}")
  #                   # Smile!
  #                   cam.snap()
  #               if state == 0: _state = "train"
  #               else: _state = "test"
  #               cam.animate().save(os.path.join(run_dir, f"({image+1})_{title}_{_state}_{self.init_time}.mp4"), writer="ffmpeg")
  #
  #       with open(os.path.join(run_dir, f"{title}_{self.init_time}.pickle"), 'wb') as f:
  #           data = {"images": self.images, "loss": self.losses, "g_truth": self.true_images}
  #           pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
