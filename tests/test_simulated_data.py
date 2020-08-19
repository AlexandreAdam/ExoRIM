from ExoRIM.simulated_data import CenteredImagesGenerator
from ExoRIM.physical_model import PhysicalModelv1
import numpy as np


def test_generator():
    N = 21
    pix = 32
    wavel = 0.5e-6
    plate_scale = 3.2
    mask = np.random.normal(0, 6, [N, 2])
    phys = PhysicalModelv1(
        pixels=pix,
        mask_coordinates=mask,
        wavelength=wavel,
        plate_scale=plate_scale,
        SNR=100
    )
    gen = CenteredImagesGenerator(phys, 100)
    for i, (X, Y) in enumerate(gen.generator()):
        pass

test_generator()