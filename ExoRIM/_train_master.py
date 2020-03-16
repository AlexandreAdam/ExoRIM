from ExoRIM.model import RIM, MSE
from ExoRIM.data_generator import SimpleGenerator
from datetime import datetime
import tensorflow as tf
import numpy as np


class TrainMaster:
    """
    This class initialize the model and utilities for visualization and training. Training and
    TrainViz inherit from this class.
    """
    def __init__(
            self,
            generator=SimpleGenerator,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=MSE(),
            model_name="RIM",
            epochs=10,
            total_images=100,
            save_percent=0.1,
            split=0.8,
            steps=12,  # number of steps for the reconstruction
            pixels=32,
            state_size=16,  # hidden state 2D size
            state_depth=2,  # Channel dimension of hidden state
            noise_std=0.1,
            num_cell_features=2,
    ):
        self.init_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
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
        self.generator = generator(total_items=total_images, split=split)
        self.loss = loss
        self.optimizer = optimizer

        # utilities for plotting and saving weigths
        self.train_traces, self.test_traces = self._pick_traces(total_images, save_percent, split)
        self.train_losses = []
        self.test_losses = []
        self.weight_grads = []

        self.trained = False

    @staticmethod
    def _pick_traces(total_images, save_percent, split):
        """
        This method selects a number of images that will be plotted and saved during training. It also picks out
        the moment during training when loss is displayed to the screen and saved for plotting.
        :param total_images: Total images in an epoch
        :param save_percent: Percentage of images to save
        :param split: Split between train and test set
        :return: Index list of image to be saved in an epoch
        """
        np.random.seed(42)
        possible_train_index = np.arange(1, int(total_images * split))
        possible_test_index = np.arange(1, int(total_images * (1 - split)))
        train_to_save = round(total_images * split * save_percent)
        test_to_save = round(total_images * (1 - split) * save_percent)
        train_traces = np.random.choice(possible_train_index, size=train_to_save, replace=False)
        test_traces = np.random.choice(possible_test_index, size=test_to_save, replace=False)
        return train_traces, test_traces
