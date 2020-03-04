import numpy as np


class DataGenerator:
    """
    Generate blurred images of point sources from a chosen psf.
    """

    def __init__(self, train_batch_size=1, test_batch_size=1, pixels=32, channels=1):
        """

        :param train_batch_size: Number of images to create for a train batch
        :param test_batch_size: Number of images to create for a test batch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param filters: Number of color measured by the camera (filter=1 is a gray image)
        """
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.pixels = pixels
        self.channels = channels

    def gaussian_psf(self, xp, yp, width):
        square_side = np.arange(self.pixels) - self.pixels / 2.
        xx, yy = np.meshgrid(square_side, square_side)
        rho_squared = (xx - xp) ** 2 + (yy - yp) ** 2
        return 1. / (width * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared/width**2))

    @staticmethod
    def normalize(vector):
        return (vector - vector.min()) / (vector.max() - vector.min())

    def noise(self):
        pass

    def tag_info(self):
        """
        Idea is to tag sep, orientation and contrast to pairs of stars generated.
        :return:
        """
        pass

    def training_batch(self):
        pass

    def test_batch(self):
        pass


class DataGenerator1(object):
    def __init__(self, train_batch_size=1, test_batch_size=1, pixels=128, filters=1):
        """

        :param train_batch_size: Number of images to create for a train batch
        :param test_batch_size: Number of images to create for a test batch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param filters: Number of color measured by the camera (filter=1 is a gray image)
        """
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.pixels = pixels
        self.filters = filters

    def train_batch(self):
        pass

    def test_batch(self):
        pass



