import numpy as np
from ExoRIM.definitions import k_truncated_poisson, dtype
import tensorflow as tf

class CenteredDataset:
    def __init__(
            self,
            total_items=1000,
            seed=42,
            split=0.8,
            train_batch_size=1,
            channels=1,
            pixels=32,
    ):
        """

        :param total_items: Total number of item to generate for an epoch
        :param train_batch_size: Number of images to create for a train batch
        :param test_batch_size: Number of images to create for a test batch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param channels: Color dimension
        :param max_number_point_sources: Maximum number of point source in a single image
        :param highest_contrast: Determine the faintest object created in an image: is defined as the delta in
                apparent magnitude
        :param save_ratio: Number between 0 and 1, percentages of individual samples to be saved
        """
        assert channels == 1
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.split = split
        self.train_batches_in_epoch = total_items * split // train_batch_size
        self.pixels = pixels
        self.total_items = total_items
        self.widths = self._widths()
        split = int(self.train_batch_size * self.train_batches_in_epoch)
        images = self.generate_epoch_images()
        self.train_set = tf.data.Dataset.from_tensor_slices(images[0:split, :, :, :])
        self.train_set = self.train_set.batch(train_batch_size)
        self.test_set = tf.data.Dataset.from_tensor_slices(images[split:, :, :, :])

    def gaussian_psf_convolution(self, sigma, xp=0, yp=0):
        """
        #TODO make the unit conversion between pixel, arcsec and meters in image plane explicit in this function
        :param xp: x coordinates of the points sources in meter (should be list of numpy array)
        :param yp: y coordinates of the points sources in meter ("")
        :param intensities: intensity of the point source, should be in watt per meter squared
        :param sigma: radius of the point source image
        :return: Image plane field, normalized
        """
        image_coords = np.arange(self.pixels) - self.pixels / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        image = np.zeros_like(xx)
        rho_squared = (xx - xp) ** 2 + (yy - yp) ** 2
        image += 1 / (sigma * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma ** 2))
        image = self.normalize(image)
        return image

    @staticmethod
    def normalize(vector):
        return (vector - vector.min()) / (vector.max() - vector.min())

    def generate_epoch_images(self):
        """

        :return: image tensor of size (total_items, pixels, pixels)
        """
        images = np.zeros(shape=(self.total_items, self.pixels, self.pixels, 1))  # one channel
        for i in range(self.total_items):
            width = self.widths[i]
            # Make this work with multiple channels
            images[i, :, :, 0] += self.gaussian_psf_convolution(sigma=width)
        return tf.convert_to_tensor(images, dtype=dtype)

    def _widths(self):
        width = np.random.uniform(3, self.pixels / 8, size=self.total_items)  # max width is arbitrary
        return width

    dataset = CenteredDataset(
            total_items=10,
            seed=42,
            split=0.8,
            train_batch_size=2,
            pixels=32)
