import numpy as np
import tensorflow as tf
from .definitions import DTYPE
from .physical_model import PhysicalModel
import math


def default_sigma_distribution(batch_size):
    return np.random.uniform(low=1e-4, high=1e-2, size=batch_size)


class CenteredBinariesDataset(tf.keras.utils.Sequence):
    def __init__(
            self,
            phys: PhysicalModel,
            total_items=1000,
            batch_size=10,
            amplitude_sigma_distribution=default_sigma_distribution,
            phase_sigma_distribution=default_sigma_distribution,
            width=2,  # sigma parameter of super gaussian
            min_separation=2,
            max_separation=None,
            seed=None
    ):
        self.seed = seed
        self.total_items = total_items
        self.pixels = phys.pixels
        self.width = width
        self.max_separation = phys.pixels/2 if max_separation is None else max_separation
        self.batch_size = batch_size
        self.phys = phys
        self.min_separation = min_separation

        self.amplitude_sigma_distribution = amplitude_sigma_distribution
        self.phase_sigma_distribution = phase_sigma_distribution

        # make coordinate system
        x = np.arange(phys.pixels) - phys.pixels//2 + 0.5 * (phys.pixels%2)
        xx, yy = np.meshgrid(x, x)
        self.x = xx
        self.y = yy

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size)

    def __getitem__(self, idx):
        return self.generate_batch(idx)

    def generate_batch(self, idx):
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        separation = np.random.uniform(size=[self.batch_size], low=self.min_separation, high=self.max_separation)
        angle = np.random.uniform(size=[self.batch_size], low=0, high=np.pi)
        images = np.zeros(shape=[self.batch_size, self.pixels, self.pixels, 1])
        for i in range(self.batch_size):
            for j in range(2): # make a 180 rotation for j=1
                x0 = separation[i] * np.cos(angle[i] + j * np.pi)/2
                y0 = separation[i] * np.sin(angle[i] + j * np.pi)/2
                images[i, ..., 0] += self.super_gaussian(x0, y0)

        images = images / images.max(axis=(1, 2), keepdims=True)
        images = tf.constant(images, dtype=DTYPE)
        amp_noise = self.amplitude_sigma_distribution(self.batch_size)
        phase_noise = self.phase_sigma_distribution(self.batch_size)
        X = self.phys.noisy_forward(images, amp_noise, phase_noise)
        return X, images

    def super_gaussian(self, x0, y0):
        rho = np.hypot(self.x - x0, self.y - y0)
        return np.exp(-0.5 * (rho/self.width)**4)


