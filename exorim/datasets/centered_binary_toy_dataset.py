import numpy as np
import tensorflow as tf
from exorim.definitions import DTYPE, LOGFLOOR
from exorim.physical_model import PhysicalModel
import math


def default_sigma_distribution(batch_size, nbuv):
    return np.ones(shape=[batch_size, nbuv]) * 1e-6


def default_contrast_distribution(batch_size):
    # log-uniform distribution
    return 10**np.random.uniform(low=-1, high=0, size=batch_size)


class CenteredBinariesDataset(tf.keras.utils.Sequence):
    def __init__(
            self,
            phys: PhysicalModel,
            total_items=1000,
            batch_size=10,
            sigma_distribution=default_sigma_distribution,
            contrast_distribution=default_contrast_distribution,
            width=2,  # sigma parameter of super gaussian
            min_separation=4,
            max_separation=None,
            seed=None,
    ):
        self.seed = seed
        self.total_items = total_items
        self.pixels = phys.pixels
        self.width = width
        self.batch_size = batch_size
        self.phys = phys
        self.max_separation = phys.pixels/2 if max_separation is None else max_separation
        self.min_separation = min_separation

        self.sigma_distribution = sigma_distribution
        self.contrast_distribution = contrast_distribution

        # make coordinate system
        x = np.arange(phys.pixels) - phys.pixels//2 + 0.5 * (phys.pixels%2)
        xx, yy = np.meshgrid(x, x)
        self.x = xx
        self.y = yy

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size)

    def __getitem__(self, idx):
        return self.generate_batch(idx)

    def __next__(self):
        return self.generate_batch(None)

    def generate_batch(self, idx):
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        separation = np.random.uniform(size=[self.batch_size], low=self.min_separation, high=self.max_separation)
        angle = np.random.uniform(size=[self.batch_size], low=0, high=2*np.pi)
        contrast = self.contrast_distribution(self.batch_size)
        images = np.zeros(shape=[self.batch_size, self.pixels, self.pixels, 1])
        for i in range(self.batch_size):
            for j in range(2): # make a 180 rotation for j=1
                x0 = separation[i] * np.cos(angle[i] + j * np.pi)/2
                y0 = separation[i] * np.sin(angle[i] + j * np.pi)/2
                images[i, ..., 0] += self.super_gaussian(1. if j == 0 else contrast[i], x0, y0)

        # images = images / images.sum(axis=(1, 2), keepdims=True)
        images = tf.maximum(images, LOGFLOOR)
        images = tf.constant(images, dtype=DTYPE)
        sigma = self.sigma_distribution(self.batch_size, self.phys.nbuv)
        X, sigma = self.phys.noisy_forward(images, sigma)
        return X, images, sigma

    def super_gaussian(self, I, x0, y0):
        rho = np.hypot(self.x - x0, self.y - y0)
        im = np.exp(-0.5 * (rho/self.width)**4)
        im /= im.sum()
        im *= I
        return im


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from exorim import PhysicalModel
    phys = PhysicalModel(128, oversampling_factor=None, logim=True)
    D = CenteredBinariesDataset(phys, 1, 1, width=2)
    x, y, sigma = D.generate_batch(0)
    x_true = phys.forward(y)

    plt.figure()
    Ainv = phys.operators.ndftm_matrix(32, phys.plate_scale, inv=True)
    dirty_beam = np.real((Ainv @ x[0, :phys.nbuv].numpy()).reshape([32, 32]))
    plt.imshow(dirty_beam, cmap="hot", norm=LogNorm(vmin=1e-6, vmax=1))
    plt.colorbar()
    plt.show()
    print(phys.chi_squared(y, x, sigma))
    print(x_true[0, :phys.nbuv])
    # print(sigma[0, phys.nbuv:])
    plt.figure()
    plt.imshow(y[0, ..., 0], cmap="hot", norm=LogNorm(vmin=1e-6, vmax=1))
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(x_true[0], "k-")
    plt.plot(x[0], "r--", lw=3)
    plt.show()

    plt.figure()
    plt.plot((x - x_true)[0]**2 / sigma[0]**2)
    plt.title(np.sum((x - x_true)[0]**2 / sigma[0]**2))
    plt.show()

