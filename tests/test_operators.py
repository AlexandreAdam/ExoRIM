from ExoRIM.operators import Baselines
from pynfft.nfft import NFFT
from ExoRIM.operators import NDFTM
from ExoRIM.definitions import mas2rad
import numpy as np
import time


def test_baselines():
    N = 11
    L = 100
    var = 10
    x = (L + np.random.normal(0, var, N)) * np.cos(2 * np.pi * np.arange(N) / N)
    y = (L + np.random.normal(0, var, N)) * np.sin(2 * np.pi * np.arange(N) / N)
    circle_mask = np.array([x, y]).T
    B = Baselines(circle_mask)

def test_pynfft():

    pixels = 128
    wavel = 1.5e-6
    plate_scale = 1.2  # mas / pixel
    x = np.arange(pixels) - pixels // 2
    xx, yy = np.meshgrid(x, x)
    image = np.zeros_like(xx)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    image = image + 1.0 * (rho < 5)

    mask = np.random.normal(0, 6, [100, 2])

    def uv_samples(mask):
        N = mask.shape[0]
        uv = np.zeros([N * (N - 1) // 2, 2])
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                uv[k, 0] = mask[i, 0] - mask[j, 0]
                uv[k, 1] = mask[i, 1] - mask[j, 1]
                k += 1
        return uv

    uv = uv_samples(mask)
    start = time.time()
    ndft = NDFTM(uv, wavel, pixels, plate_scale)
    vis1 = np.einsum("ij, j -> i", ndft, np.ravel(image))
    end = time.time() - start
    print(f"Took {end:.4f} second to compute NDFTM")

    # TODO implement this in new physical model
    phase = np.exp(-1j * np.pi / wavel * mas2rad(plate_scale) * (uv[:, 0] + uv[:, 1]))
    start = time.time()
    plan = NFFT([pixels, pixels], uv.shape[0], n=[pixels, pixels])
    plan.x = uv / wavel * mas2rad(plate_scale)
    plan.f_hat = image
    plan.precompute()
    plan.trafo()
    vis = plan.f.copy() * phase
    end = time.time() - start
    print(f"Took {end:.4f} seconds to compute NFFT")
    assert np.allclose(np.abs(vis), np.abs(vis1), rtol=1e-5)
    assert np.allclose(np.sin(np.angle(vis) - np.angle(vis1)), np.zeros_like(vis), atol=1e-5)

