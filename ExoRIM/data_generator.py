import numpy as np
import astropy.units as u
from ExoRIM.definitions import datadir, k_truncated_poisson
from ExoRIM.model import PhysicalModel
import os

# Eventually, we could use to tagg information and produce more complicated cases
# TODO add tagged information to images to keep trace of contrast, separation, orientation
class DataGenerator:
    """
    Generate blurred images of point sources from a chosen psf. The idea of this generator is to produce, ideally,
    independent observations batches for training and most importantly for testing. This is why a population of samples
    defined at initiation time. We pick batches randomly in the population without replacement.


    psf: point spread function
    """
    def __init__(
            self,
            #fov=4.33 * u.arcmin, # Wide Infra-Red Camera
            #pixel_scale=0.2487 * u.arcsec / u.pixel,
            #lam=1.65 * u.microns, # H-Band,
            #diameter=5.1 * u.m, # Diameter of the aperture at Palomar # TODO determine what to do w/ these
            total_items=1000,
            split=0.8,
            train_batch_size=1,
            test_batch_size=1,
            channels=1,
            max_number_point_sources=2,
            highest_contrast=5,
            pixels=None,
            reset_each_epoch=False,
            save=0.1):
        """
        # TODO eventually add number of point sources, but our main test case is max point sources = 2
        # TODO eventually permit more than one channel, but for now only gray images are supported
        Note that if pixels defined, then we ignore pixel_scale and inversely.

        The situations that interest us are images where a relatively bright star or 2 are accompanied by faint companions
        (either brown dwarfs or super Jupiter) at close separation (~ lambda/D). This is the wheel house of the NRM
        technique. Above few lambda/D (depend on Strehl ratio), direct detections methods are better.
        So our goal is to prove that the model can reconstruct an image where the companion is identifiable for detection.

        Typical high contrast are of the order of a few delta magnitude (up to 6-7 ~ 200-300:1 contrasts).

        # Note, seeing for Palomar WFIC are overly sampled at 0.25 arcsec. #TODO myabe consider this is physical model

        :param fov: Field of view of the image, defined in arcsec
        :param total_items: Total number of item to generate for an epoch
        :param train_batch_size: Number of images to create for a train batch
        :param test_batch_size: Number of images to create for a test batch
        :param pixels: Number of pixels on a side of the square CCD camera
        :param channels: Color dimension
        :param max_number_point_sources: Maximum number of point source in a single image
        :param highest_contrast: Determine the faintest object created in an image: is defined as the delta in
                apparent magnitude
        :param pixelscale: Angular size of a single pixel
        :param reset_each_epoch: Whether to create new population each epoch (best practice should be False?)
        :param save: Number between 0 and 1, percentages of individual samples to be saved
        """
        assert max_number_point_sources < 3 and max_number_point_sources > 0
        assert channels == 1
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Number of batches in an epoch
        self.train_batches_in_epoch = total_items*split // train_batch_size
        self.test_batches_in_epoch = total_items*(1 - split) // test_batch_size

        #self.res = self.spatial_resolution(lam, diameter) # in arcsec
        self.highest_contrast = highest_contrast
        self.pixels = pixels
        self.channels = channels
        self.max_point_source = max_number_point_sources
        self.total_items = total_items

    def gaussian_psf_convolution(self, xp, yp, intensities, sigma):
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
        for i, _ in enumerate(xp):
            rho_squared = (xx - xp[i])**2 + (yy - yp[i])**2
            image += intensities[i] / (sigma[i] * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma[i]**2))
        image = self.normalize(image)
        # TODO think about where to put this normalisation, is the convolution the right place?
        return image

    # @staticmethod
    # def spatial_resolution(lam, d):
    #     return (1.22 * lam / d * u.deg).to(u.arcsec)

    @staticmethod
    def normalize(vector):
        return (vector - vector.min()) / (vector.max() - vector.min())

    def nps(self, p="uniform", mu=2):
        """
        A lam = 2 with poisson distribution is interesting for modeling binary systems.

        :param p: Probability of choosing k sources in an image.
            Default is uniform: each k between 1 and maximum point sources are eually likely.
            Poisson: probability taken from Poisson distribution with expectated value and variance mu.
        :param mu: Expected value of the poisson distribution if chosen
        :return: number of point source for each image in an epoch
        """
        pool = np.arange(1, self.max_point_source, dtype=np.int32)
        if p == "uniform":
            p = None
        elif p == "poisson":
            p = k_truncated_poisson(k=pool, mu=mu)
        else:
            raise NotImplementedError(f"p={p} but supported are uniform and poisson")

        return np.random.choice(pool, size=self.total_items, p=p)

    def contrast(self):
        """
        We are interested in situations where faint object ar present in the vicinity of one or two bright objects.
        :return:
        """
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

# This generator is a simpler version of the previous one, to be used for testing the pipeline
# It produces a random number of gaussian blobs to be reconstructed by the model
class SimpleGenerator:
    """ #TODO relate width of the PSF to intensity -- otherwise unphysical
    # TODO possibly makes this work for multiple channels?
     Generate blurred images of point sources from a chosen psf. The idea of this generator is to produce, ideally,
     independent observations batches for training and most importantly for testing. This is why a population of samples
     defined at initiation time. We pick batches randomly in the population without replacement.


     psf: point spread function
     """

    def __init__(
            self,
            total_items=1000,
            split=0.8,
            train_batch_size=1,
            test_batch_size=1,
            channels=1,
            max_number_point_sources=10,
            highest_contrast=5,
            pixels=32,
            save_ratio=0.1
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
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Number of batches in an epoch
        self.train_batches_in_epoch = total_items * split // train_batch_size
        self.test_batches_in_epoch = total_items * (1 - split) // test_batch_size
        self.train_index = 0 # index to keep track of progress in an epoch
        self.test_index = 0

        self.highest_contrast = highest_contrast
        self.pixels = pixels
        self.channels = channels
        self.max_point_source = max_number_point_sources
        self.total_items = total_items
        self.save_ratio = save_ratio
        self.coordinates = self._coordinates()
        self.contrast = []
        self.widths = []
        self.images = self.generate_epoch_images()

    def gaussian_psf_convolution(self, xp, yp, intensities, sigma):
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
        for i, _ in enumerate(xp):
            rho_squared = (xx - xp[i])**2 + (yy - yp[i])**2
            image += intensities[i] / (sigma[i] * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma[i] ** 2))
        image = self.normalize(image)
        return image

    @staticmethod
    def normalize(vector):
        return (vector - vector.min()) / (vector.max() - vector.min())

    def generate_epoch_images(self):
        """

        :return: image tensor of size (total_items, pixels, pixels)
        """
        images = np.zeros(shape=(self.total_items, self.pixels, self.pixels, 1)) # one channel
        for i in range(self.total_items):
            xp = self.coordinates[i][:, 0]
            yp = self.coordinates[i][:, 1]
            intensities = self._intensities(nps=xp.size)  # TODO intensities should be simpler
            width = self._widths(nps=xp.size)
            # Make this work with multiple channels
            images[i, :, :, 0] += self.gaussian_psf_convolution(xp=xp, yp=yp, intensities=intensities, sigma=width)
        return images

    def _nps(self, p="poisson", mu=2):
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

    def _widths(self, nps):
        width = np.random.uniform(2, self.pixels / 6, size=nps) # max width is arbitrary
        self.widths.append(width) #TODO make this more readable from init
        return width

    def _intensities(self, nps):
        """
        Given a number of point sources, pick normalised intensities with first source as the brightest
        :param nps: number of point sources
        :return: intensities list
        """
        intensities = [1]
        contrast = np.zeros(nps)
        if nps > 1:
            contrast[1:] = np.random.uniform(0.1, self.highest_contrast, size=nps - 1)
            self.contrast.append(contrast) #TODO make this more readable from init
            flux = 100**(-contrast / 5)
            intensities += list(flux)
        return intensities

    def _coordinates(self):
        """
        index 0: xp
        index 1: yp
        :return: list of coordinate arrays
        """
        coords = []
        pool = np.arange(-self.pixels // 2, self.pixels // 2 + 1)
        for nps in self._nps():
            coords.append(np.random.choice(pool, size=(nps, 2)))
        return coords

    def training_batch(self):
        while self.train_index <= self.train_batches_in_epoch:
            li = self.train_index * self.train_batch_size # lower index of batch
            ui = li + self.train_batch_size  # upper index of batch
            yield self.images[li:ui, :, :, :]
            self.train_index += 1
        self.train_index = 0  # reset after epoch

    def test_batch(self):
        while self.test_index <= self.test_batch_size:
            split = int(self.train_batch_size * self.train_batches_in_epoch)
            li = split + self.test_index * self.test_batch_size
            ui = li + self.test_batch_size
            yield self.images[li:ui, :, :, :]
            self.test_index += 1
        self.test_index = 0
