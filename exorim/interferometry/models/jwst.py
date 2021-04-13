from exorim.interferometry.operators import NDFTM, \
    closure_fourier_matrices, Baselines, \
    closure_phase_operator, closure_phase_covariance_inverse
from scipy.signal import get_window
from exorim.definitions import MYCOMPLEX, DTYPE
import numpy as np
from astropy.io import fits
import astropy.units as u
import tensorflow as tf
import os

_here = os.path.dirname(__file__)
JWST_NIRISS_MASK = np.array([ #V2/m and V3/m coordinates
    [ 1.143,  1.980],  # C1
    [ 2.282,  1.317],  # B2
    [ 2.286,  0.000],  # C2
    [ 0.000, -2.635],  # B4
    [-2.282, -1.317],  # B5
    [-2.282,  1.317],  # B6
    [-1.143,  1.980]   # C6
])


class PhysicalModel:
    def __init__(self,
                 pixels,
                 loglikelihood="append_visibility_amplitude_closure_phase",
                 window="hamming",
                 temperature=1,
                 filter="F380M",
                 SNR=100,
                 vis_phase_std=0.1,
                 logim=True,
                 rotations=False,  # assume we have done mask rotations or not. False correspond to a single observation
                 psf="jwst_F430M_2_psf.fits",
                 lam=0):  # regularization
        assert loglikelihood in ["append_visibility_amplitude_closure_phase", "visibilities", "visibility_amplitude"]
        self.pixels = pixels  # number of pixels of the image output of the model -> to be zero-padded to psf size
        self.filter = filter
        self.baselines = Baselines(mask_coordinates=JWST_NIRISS_MASK)
        _fits = fits.open(os.path.join(_here, "psf", psf))
        psf = _fits[0].data
        h = get_window(window, psf.hape[0])
        window = np.outer(h, h)
        self.platescale = _fits[0].header["PIXELSCL"]/1000       # mas
        self.wavelength = _fits[0].header["WAVELEN"]             # meters
        self.fov = _fits[0].header["FOV"]/1000                   # mas
        self.diffraction_limit = _fits[0].header["DIFLMT"]/1000  # mas


        self.A = NDFTM(self.baselines.UVC, self.wavelength, psf.hape[0], self.platescale)




