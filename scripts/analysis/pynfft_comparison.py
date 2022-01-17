from pynfft.nfft import NFFT
from exorim.operators import NDFTM
from exorim.definitions import mas2rad
import numpy as np
from scipy.special import j1
import time
import matplotlib.pyplot as plt

pixels = 128
wavel = 1.5e-6
plate_scale = 1.2  # mas / pixel
x = np.arange(pixels) - pixels//2
xx, yy = np.meshgrid(x, x)
image = np.zeros_like(xx)
rho = np.sqrt(xx**2 + yy**2)
image = image + 1.0 * (rho < 10)
plt.imshow(image)
plt.show()
mask = np.random.normal(0, 6, [100, 2])


def uv_samples(mask):
    N = mask.shape[0]
    uv = np.zeros([N * (N - 1)//2, 2])
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            uv[k, 0] = mask[i, 0] - mask[j, 0]
            uv[k, 1] = mask[i, 1] - mask[j, 1]
            k += 1
    return uv


def pulse(x, y, pdim):
    rm = 0.5 * pdim
    return j1(rm * np.sqrt(x**2 + y**2)) / np.sqrt(x**2 + y**2) / rm**2


uv = uv_samples(mask)
start = time.time()
ndft = NDFTM(uv, wavel, pixels, plate_scale)
vis1 = np.einsum("ij, j -> i", ndft, np.ravel(image))
end = time.time() - start
print(f"Took {end:.4f} second to compute NDFTM")

m2pix = mas2rad(plate_scale) * pixels / wavel
phase = np.exp(-1j * np.pi / wavel * mas2rad(plate_scale) * (uv[:, 0] + uv[:, 1]))
start = time.time()
plan = NFFT([pixels, pixels], uv.shape[0], n=[pixels, pixels])
plan.x = uv / wavel * mas2rad(plate_scale)
plan.f_hat = image
plan.precompute()
plan.trafo()
vis = plan.f.copy() * phase
end = time.time() - start
print(f"Took {end:.4f} seconds to compute NFFT") # usually at least 5x faster, scales better with large number of pixels
# print(np.abs(vis))
# print(np.abs(vis1))
plt.figure()
plt.plot(sorted(np.abs(vis)), color="k")
plt.plot(sorted(np.abs(vis1)), color="r")
plt.show()
plt.figure()
plt.plot(sorted(np.angle(vis)), color="k")
plt.plot(sorted(np.angle(vis1)), color="r")
plt.show()
assert np.allclose(np.abs(vis), np.abs(vis1), rtol=1e-3)
assert np.allclose(np.sin(np.angle(vis) - np.angle(vis1)), np.zeros_like(vis), atol=1e-3)
# print(np.angle(vis))

