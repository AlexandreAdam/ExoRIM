from ExoRIM.simulated_data import CenteredImagesGenerator
from ExoRIM.physical_model import PhysicalModelv1
import numpy as np


def test_generator():
    N = 21
    pix = 64
    wavel = 0.5e-6
    mask = np.random.normal(0, 6, [N, 2])
    phys = PhysicalModelv1(
        pixels=pix,
        mask_coordinates=mask,
        wavelength=wavel,
        SNR=100
    )
    gen = CenteredImagesGenerator(phys, 100)
    for i, (X, Y) in enumerate(gen.generator()):
        pass

if __name__ == '__main__':
    test_generator()