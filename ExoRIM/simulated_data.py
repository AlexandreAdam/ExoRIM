import numpy as np
import tensorflow as tf
from ExoRIM.definitions import k_truncated_poisson, centroid, dtype
from ExoRIM.physical_model import PhysicalModel


class CenteredImagesv1:
    def __init__(
            self,
            total_items=1000,
            seed=42,
            channels=1,
            pixels=32,
            highest_contrast=0.5,
            max_point_sources=10
    ):
        """
        This class defines the characteristics of a simulated dataset to train ExoRIM. It produces images with a
        centered blob and few fainter point sources near the center of the image. Information like widths of the
        point sources and contrasts are attached to the object for future analysis and reproducibility.

        :param total_items: Total number of item to generate for an epoch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param channels: Color dimension
        :param max_point_sources: Maximum number of point source in a single image
        :param highest_contrast: Determine the faintest object created in an image: is defined as the delta in
                apparent magnitude
        """
        assert channels == 1
        np.random.seed(seed)
        self.pixels = pixels
        self.total_items = total_items

        self.highest_contrast = highest_contrast
        self.max_point_source = max_point_sources
        self.nps = self._nps()
        self.widths = self._widths()
        self.coordinates = self._coordinates()
        self.contrast = self._contrasts()

    def gaussian_psf_convolution(self, sigma, intensity,  xp=0, yp=0):
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
        image += intensity / (sigma * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma ** 2))
        return image

    def circular_psf(self, sigma, intensity, xp, yp):
        image_coords = np.arange(self.pixels) - self.pixels / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        image = np.zeros_like(xx)
        rho = np.sqrt((xx - xp) ** 2 + (yy - yp) ** 2)
        image += intensity * (rho < sigma)
        return image

    @staticmethod
    def normalize(X, minimum, maximum):
        return minimum + (X - X.min()) * (maximum - minimum) / (X.max() - X.min() + 1e-8)

    def generate_epoch_images(self):
        """

        :return: image tensor of size (total_items, pixels, pixels)
        """
        images = np.zeros(shape=(self.total_items, self.pixels, self.pixels, 1))  # one channel
        # images += 1e-5 * np.random.random(size=images.shape) + 1e-4  # background
        for i in range(self.total_items):
            for j, _ in enumerate(range(self.nps[i])):
                images[i, :, :, 0] += self.gaussian_psf_convolution(
                    self.widths[i][j],
                    1 - self.contrast[i][j],
                    *self.coordinates[i][j]
                )
        # normalize
        images = self.normalize(images, minimum=0, maximum=1)
        # recenter image
        for j, im in enumerate(images[...,0]):
            (x0, y0) = centroid(im, threshold=1e-4)
            dy = int(y0) - self.pixels//2
            dx = int(x0) - self.pixels//2
            im = np.pad(im, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), constant_values=(0, 0))
            im = np.roll(np.roll(im, -dx, axis=1), -dy, axis=0)
            cutdy = -abs(dy) if dy != 0 else None
            cutdx = -abs(dx) if dx != 0 else None
            im = im[abs(dy):cutdy, abs(dx):cutdx]
            images[j, ..., 0] = im
        return images

    def _nps(self, p="uniform", mu=2):
        """
        A mu = 2 with poisson distribution is interesting for modeling mostly binary systems.

        :param p: Probability of choosing k sources in an image.
            Default is uniform: each k between 1 and maximum point sources are equally likely.
            Poisson: probability taken from Poisson distribution with expectated value and variance mu.
        :param mu: Expected value of the poisson distribution if chosen
        :return: number of point source for each image in an epoch
        """
        pool = np.arange(2, self.max_point_source + 1)
        if p == "uniform":
            p = None
        elif p == "poisson":
            p = k_truncated_poisson(k=pool, mu=mu)
        else:
            raise NotImplementedError(f"p={p} but supported are uniform and poisson")
        return np.random.choice(pool, size=self.total_items, p=p)

    def _central_widths(self):
        width = np.random.uniform(3, 5, size=self.total_items)
        return width

    def _widths(self):
        widths = [[] for _ in range(self.total_items)]
        for i, nps in enumerate(self.nps):
            widths[i].extend(np.random.uniform(1, 4, size=nps).tolist())
        return widths

    def _contrasts(self):
        """
        Given a number of point sources, chose a contrast between point sources and central object
        :return: contrast list for each image [[0, 0.4, ...]]
        """
        contrasts = [[0] for _ in range(self.total_items)]
        for i, nps in enumerate(self.nps):
            contrasts[i].extend(np.random.uniform(0.1, self.highest_contrast, size=nps).tolist())
        return contrasts

    def _coordinates(self):
        """
        :return: list of coordinate arrays [[(xp0, yp0), ... (xpn, ypn)], ...]
        """
        coords = [[] for _ in range(self.total_items)]
        # coordinate in central square (5 pixels from the edge) to avoid objects at the edge.
        pool = np.arange(-self.pixels // 2 + 8, self.pixels // 2 - 8)
        for i, nps in enumerate(self.nps):
            coords[i].extend(np.random.choice(pool, size=(nps, 2)).tolist())
        return coords


class OffCenteredBinaries:
    """
    This class create the meta data for a dataset composed of randomly placed binary objects (gaussian blob).
    """
    def __init__(
            self,
            total_items=1000,
            seed=42,
            channels=1,
            pixels=32,
            highest_contrast=0.3,
            max_distance=8  # in pixels
    ):
        np.random.seed(seed)
        self.total_items = total_items
        self.pixels = pixels
        self.highest_contrast = highest_contrast
        self.max_distance = max_distance

        self.distance = self._distance()
        self.angle = self._angle()
        self.intensities = self._intensities()

    def gaussian_psf_convolution(self, sigma, intensity, xp=0, yp=0):
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
        image += intensity / (sigma * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma ** 2))
        image = self.normalize(image)
        return image

    def generate_epoch_images(self):
        """

        :return: image tensor of size (total_items, pixels, pixels)
        """
        images = np.zeros(shape=(self.total_items, self.pixels, self.pixels, 1))  # one channel
        images += 1e-4 * np.random.random(size=images.shape) + 1e-5 # background
        for i in range(self.total_items):
            for j in range(2):
                sign = 1 if j == 0 else -1
                x = int(np.ceil(sign * self.distance[i]/2 * np.cos(self.angle[i]))) + self.pixels//2
                y = int(np.ceil(sign * self.distance[i]/2 * np.sin(self.angle[i]))) + self.pixels//2
                images[i, x, y, 0] += self.intensities[i][j]
        return images

    @staticmethod
    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())

    def _angle(self):
        pool = np.linspace(-np.pi, np.pi, 1000)
        return np.random.choice(pool, size=self.total_items)

    def _distance(self):
        pool = np.arange(2, self.max_distance)
        return np.random.choice(pool, size=self.total_items)

    def _intensities(self):
        """

        :return: list of intensity of the center point [[1, 1 - contrast]]
        """
        contrast = np.random.random(size=self.total_items) * self.highest_contrast
        return [[1, 1 - c] for c in contrast]


class CenteredCircle:
    def __init__(
            self,
            total_items=1000,
            seed=42,
            channels=1,
            pixels=32,
    ):
        """
        This class defines the characteristics of a simulated dataset to train ExoRIM. It produces images with a
        centered blob.

        :param total_items: Total number of item to generate for an epoch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param channels: Color dimension
        """
        assert channels == 1
        np.random.seed(seed)
        self.pixels = pixels
        self.total_items = total_items
        self.widths = self._widths()

    def generate_epoch_images(self):
        """

        :return: image tensor of size (total_items, pixels, pixels)
        """
        image_coords = np.arange(self.pixels) - self.pixels / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        images = np.zeros(shape=(self.total_items, self.pixels, self.pixels, 1))  # one channel
        for i in range(self.total_items):
            images[i, :, :, 0] += rho < self.widths[i]
        # normalize by flux
        images = images #/ np.reshape(np.sum(images, axis=(1, 2, 3)), [images.shape[0], 1, 1, 1])
        return images

    def _widths(self):
        return np.random.uniform(0, 12, size=self.total_items)


class CenteredImagesGenerator:
    def __init__(
            self,
            physical_model: PhysicalModel,
            total_items_per_epoch,
            channels=1,
            pixels=32,
            highest_contrast=0.3,
            max_point_sources=10,
            save=None,  # should be the data directory path already created!
            fixed=False
    ):
        self.physical_model = physical_model
        self.total_items_per_epoch = total_items_per_epoch
        self.channels = channels
        self.pixels = pixels
        self.highest_contrast = highest_contrast
        self.max_point_sources = max_point_sources
        self.epoch = -1  # internal variable to reseed the random generator each epoch if fixed is false
        self.save = save
        # fixed switch allows train on the same dataset each epoch and reproduce it if needed with the seed 42
        self.fixed = fixed

    def generator(self):
        self.epoch += 1
        if not self.fixed:
            np.random.seed(self.epoch)
        else:
            np.random.seed(42)
        for i in range(self.total_items_per_epoch):
            Y = tf.constant(self.generate_image(), dtype=dtype)
            X = self.physical_model.simulate_noisy_data(tf.reshape(Y, [1, *Y.shape]))
            X = tf.reshape(X, X.shape[1:])  # drop the batch dimension acquired in physical model
            yield X, Y

    def gaussian_psf_convolution(self, image, sigma, intensity,  xp=0, yp=0):
        image_coords = np.arange(image.shape[0]) - image.shape[0] / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        image = np.zeros_like(xx)
        rho_squared = (xx - xp) ** 2 + (yy - yp) ** 2
        image += intensity * np.exp(-0.5 * (rho_squared / sigma ** 2))
        return image

    def circular_psf(self, sigma, intensity, xp, yp):
        image_coords = np.arange(self.pixels) - self.pixels / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        image = np.zeros_like(xx)
        rho = np.sqrt((xx - xp) ** 2 + (yy - yp) ** 2)
        image += intensity * (rho < sigma)
        return image

    @staticmethod
    def normalize(X, minimum, maximum):
        # return np.tanh(X)
        return minimum + (X - X.min()) * (maximum - minimum) / (X.max() - X.min() + 1e-8)

    def recenter(self, image):
        (x0, y0) = centroid(image, threshold=1e-4)
        dy = int(y0) - self.pixels//2
        dx = int(x0) - self.pixels//2
        im = np.pad(image, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), constant_values=(0, 0))
        im = np.roll(np.roll(im, -dx, axis=1), -dy, axis=0)
        cutdy = -abs(dy) if dy != 0 else None
        cutdx = -abs(dx) if dx != 0 else None
        im = im[abs(dy):cutdy, abs(dx):cutdx]
        return im

    def generate_image(self):
        image = np.zeros(shape=(self.pixels, self.pixels))
        # images += 1e-5 * np.random.random(size=images.shape) + 1e-4  # background
        nps = self._nps()[0]
        width = self._width(nps)
        intensity = 1 - self._contrasts(nps)
        coordinates = self._coordinates(nps)
        for i in range(nps):
            # image += self.gaussian_psf_convolution(width[i], intensity[i], *coordinates[i])
            image += self.gaussian_psf_convolution(image, width[i], intensity[i], *coordinates[i])
        # image = self.recenter(image)
        image = self.normalize(image, minimum=0, maximum=1)
        image = np.reshape(image, newshape=[self.pixels, self.pixels, 1])
        return image

    def _nps(self, p="uniform", mu=2):
        pool = np.arange(2, self.max_point_sources + 1)
        if p == "uniform":
            p = None
        elif p == "poisson":
            p = k_truncated_poisson(k=pool, mu=mu)
        else:
            raise NotImplementedError(f"p={p} but supported are uniform and poisson")
        return np.random.choice(pool, size=1, p=p)

    def _width(self, nps):
        return np.random.uniform(1, self.pixels/3, size=nps)

    def _contrasts(self, nps):
        return np.random.uniform(0.1, self.highest_contrast, size=nps)

    def _coordinates(self, nps):
        # coordinate in central square (5 pixels from the edge) to avoid objects at the edge.
        pool = np.arange(-self.pixels // 2 + 5, self.pixels // 2 - 5)
        return np.random.choice(pool, size=(nps, 2))
