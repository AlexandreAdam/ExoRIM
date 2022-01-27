from tensorflow_nufft import nufft
import numpy as np
from exorim.physical_model import JWST_NIRISS_MASK
from exorim.operators import Operators
from exorim.definitions import rad2mas, mas2rad
import time

wavel = 3.8e-6
pixels = 128


x = np.arange(pixels) - pixels//2
xx, yy = np.meshgrid(x, x)
image = np.zeros_like(xx)
rho = np.sqrt(xx**2 + yy**2)
image = image + 1.0 * (rho < 10)


B = Operators(JWST_NIRISS_MASK, wavel)
rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2) / wavel  # frequency in 1/RAD
fov = rad2mas(1/rho).max()
resolution = (rad2mas(1/2/rho)).min()  # Michelson criterion = lambda / 2b radians
oversampling_factor = pixels * resolution / 10 / fov
plate_scale = resolution / oversampling_factor

A, A1, A2, A3 = B.build_operators(pixels, plate_scale)
start = time.time()
V = A @ image.ravel()
V1 = A1 @ image.ravel()
V2 = A2 @ image.ravel()
V3 = A3 @ image.ravel()
end = time.time() - start
print(f"Took {end:.4f} seconds to compute all DFT")
print(V)

# TODO figure out how to set up points correctly, in the range [-pi, pi]
# start = time.time()
# uv = B.UVC
# phase = np.exp(-2j * np.pi / wavel * mas2rad(plate_scale) * (uv[:, 0] + uv[:, 1]))
# points = uv / uv.max() * np.pi
# print(points)
# V = nufft(image, points) * phase
# end = time.time() - start
# print(f"Took {end:.4f} seconds to compute NUFFT")
#
# print(V)
