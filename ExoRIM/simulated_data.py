import numpy as np
from ExoRIM.definitions import k_truncated_poisson


class CenteredImagesv1:
    def __init__(
            self,
            total_items=1000,
            seed=42,
            channels=1,
            pixels=32,
            highest_contrast=0.95,
            max_point_sources=5
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
        self.widths = self._widths()

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
            # central object
            images[i, :, :, 0] += self.gaussian_psf_convolution(self.widths[i][0], 1)
            for j, _ in enumerate(range(1, self.nps[i]+1)):
                images[i, :, :, 0] += self.gaussian_psf_convolution(
                    self.widths[i][j],
                    1 - self.contrast[i][j],
                    *self.coordinates[i][j]
                )
            images[i, :, :, 0] = self.normalize(images[i, :, :, 0])
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
        pool = np.arange(1, self.max_point_source + 1)
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
        central_width = self._central_widths()
        widths = [[] for _ in range(self.total_items)]
        for i, nps in enumerate(self.nps):
            widths[i].append(central_width[i])
            widths[i].extend(np.random.uniform(1, 4, size=nps).tolist())
        return widths

    def _contrasts(self):
        """
        Given a number of point sources, chose a contrast between point sources and central object
        :return: contrast list for each image [[0, 0.4, ...]]
        """
        contrasts = [[0] for _ in range(self.total_items)]# central object contrast is 0 by definition
        for i, nps in enumerate(self.nps):
            contrasts[i].extend(np.random.uniform(0.1, self.highest_contrast, size=nps).tolist())
        return contrasts

    def _coordinates(self):
        """
        :return: list of coordinate arrays [[(xp0, yp0), ... (xpn, ypn)], ...]
        """
        coords = [[(0, 0)] for _ in range(self.total_items)] # initialize with central object
        # coordinate in central square (5 pixels from the edge) to avoid objects at the edge.
        pool = np.arange(-self.pixels // 2 + 5, self.pixels // 2 - 5)
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
            highest_contrast=0.95,
            max_width=5  # in pixels
    ):
        np.random.seed(seed)
        self.total_items = total_items
        self.pixels = pixels
        self.highest_contrast = highest_contrast
        self.max_width = max_width

        self.coordinates = self._coordinates()
        self.widths = self._widths()
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
        for i in range(self.total_items):
            for j in range(2):
                images[i, :, :, 0] += self.gaussian_psf_convolution(
                    self.widths[i][j],
                    self.intensities[i][j],
                    *self.coordinates[i][j]
                )
            images[i, :, :, 0] = self.normalize(images[i, :, :, 0])
        return images

    @staticmethod
    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())
    
    def _coordinates(self):
        """
        :return: list of coordinate arrays [[(xp0, yp0), (xp1, yp1)], ...]
        """
        coords = [[] for _ in range(self.total_items)]
        max_pixel = self.pixels//2 - self.max_width - 3
        # Leave a pad of 3 black pixels to remove information at the edge of the picture
        pool = np.arange(-max_pixel, max_pixel)
        for i in range(self.total_items):
            coords[i].extend(np.random.choice(pool, size=(2, 2)).tolist())
        return coords
    
    def _widths(self):
        """
        :return: list of widths arrays [[w0, w1], ...]
        """
        widths = [[] for _ in range(self.total_items)]
        for i in range(self.total_items):
            widths[i].append(np.random.uniform(1, self.max_width, size=2).tolist())
        return widths

    def _intensities(self):
        """

        :return: list of intensity of the center point [[1, 1 - contrast]]
        """
        contrast = np.random.random(size=self.total_items) * self.highest_contrast
        return [[1, 1 - c] for c in contrast]
