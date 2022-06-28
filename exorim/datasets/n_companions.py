import numpy as np
import tensorflow as tf
from exorim.definitions import DTYPE, LOGFLOOR
from exorim.physical_model import PhysicalModel
import math
from scipy.stats import poisson


def sigma_distribution(batch_size, nbuv):
    return np.ones(shape=[batch_size, nbuv]) * 1e-7


def n_distribution(batch_size):
    return np.atleast_1d(poisson.rvs(1, batch_size))


class NCompanions(tf.keras.utils.Sequence):
    def __init__(
            self,
            phys: PhysicalModel,
            total_items=1000,
            batch_size=10,
            sigma_distribution=sigma_distribution,
            n_distribution=n_distribution,
            width=2,  # sigma parameter of super gaussian
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
        self.min_separation = 2 * width

        self.sigma_distribution = sigma_distribution
        self.n_distribution = n_distribution

        # make coordinate system
        x = np.arange(phys.pixels) - phys.pixels//2 + 0.5 * (phys.pixels % 2 == 0)
        self.x, self.y = np.meshgrid(x, x)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size)

    def __getitem__(self, idx):
        return self.generate_batch(idx)

    def __next__(self):
        return self.generate_batch(None)

    def generate_batch(self, idx):
        ns = self.n_distribution(batch_size=self.batch_size)
        images = np.zeros(shape=[self.batch_size, self.pixels, self.pixels, 1])
        images[..., 0] += self.super_gaussian(1., 0., 0.)
        for j in range(self.batch_size):
            n = ns[j]
            if n == 0:
                continue
            separation = np.random.uniform(size=[n], low=self.min_separation, high=self.max_separation)
            angle = np.random.uniform(size=[n], low=0, high=2 * np.pi)
            contrast = 10**np.random.uniform(size=[n], low=-4, high=0)  # detection limit set at 10**(-4) for JWST NIRISS AMI
            for i in range(n):
                x0 = separation[i] * np.cos(angle[i] + j * np.pi)/2
                y0 = separation[i] * np.sin(angle[i] + j * np.pi)/2
                images[j, ..., 0] += self.super_gaussian(contrast[i], x0, y0)  # place a companion
        images = tf.constant(images, DTYPE)
        # images = images / tf.reduce_sum(images, axis=(1, 2), keepdims=True)
        images = tf.maximum(images, LOGFLOOR)
        sigma = self.sigma_distribution(self.batch_size, self.phys.nbuv)
        X, sigma = self.phys.noisy_forward(images, sigma)
        return X, images, sigma

    def super_gaussian(self, I, x0, y0):
        rho = tf.sqrt((self.x - x0)**2 + (self.y - y0)**2)
        im = np.exp(-0.5 * (rho/self.width)**4)
        im /= im.sum()
        im *= I
        return im


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from exorim import PhysicalModel
    phys = PhysicalModel(127, oversampling_factor=None, logim=False, plate_scale=None)
    D = NCompanions(phys, 1, 1, width=2)
    x, y, sigma = next(D)
    x_true = phys.forward(y)
    extent = [*[-phys.plate_scale * phys.pixels / 2, phys.plate_scale * phys.pixels / 2]*2]
    print(phys.plate_scale)

    plt.figure()
    Ainv = phys.operators.ndftm_matrix(32, phys.plate_scale, inv=True)
    dirty_beam = np.real((Ainv @ x[0, :phys.nbuv].numpy()).reshape([32, 32]))
    dirty_beam = tf.maximum(dirty_beam, LOGFLOOR)
    plt.imshow(dirty_beam, cmap="hot", norm=LogNorm(vmin=1e-6, vmax=1), extent=extent)
    plt.colorbar()
    plt.show()
    print(phys.chi_squared(y, x, sigma))
    print(x_true[0, :phys.nbuv])
    # print(sigma[0, phys.nbuv:])
    plt.figure()
    plt.imshow(y[0, ..., 0], cmap="hot", norm=LogNorm(vmin=1e-6, vmax=1), extent=extent)
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

