from ExoRIM.operators import Baselines
from pynfft.nfft import NFFT
from ExoRIM.operators import NDFTM, closure_phase_covariance, closure_phase_operator, closure_baselines_projectors
from ExoRIM.definitions import mas2rad
from scipy.special import jv
import numpy as np
import time


def test_baselines():
    N = 11
    L = 100
    var = 10
    x = (L + np.random.normal(0, var, N)) * np.cos(2 * np.pi * np.arange(N) / N)
    y = (L + np.randofm.normal(0, var, N)) * np.sin(2 * np.pi * np.arange(N) / N)
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


# def test_fourier_transform_matrix():
#     N = 12
#     mask = np.random.normal(0, 1, N)
#     B = Baselines(mask)

def test_closure_phase_covariance_operator():
    N = 4
    mask = np.random.normal(0, 2, (N, 2))
    B = Baselines(mask)
    sigma = 1
    CPO = closure_phase_operator(B)
    pc_cov = closure_phase_covariance(CPO, sigma)
    print(pc_cov)
    expected_result = np.array([
        [3, 1, -1],
        [1, 3, 1],
        [-1, 1, 3]
    ])
    assert np.all(np.equal(pc_cov, expected_result))
    sigma = 2
    CPO = closure_phase_operator(B)
    pc_cov = closure_phase_covariance(CPO, sigma)
    expected_result *= sigma**2
    print(pc_cov)
    assert np.all(np.equal(pc_cov, expected_result))


def test_closure_baseline_projectors():
    N = 5
    mask = np.random.normal(0, 2, (N, 2))
    B = Baselines(mask)
    CPO = closure_phase_operator(B)
    V1, V2, V3 = closure_baselines_projectors(CPO)
    print(V1)
    print(V2)
    print(V3)
    """
    In general, each row of Vi contains a single non-zero entry = 1, which select a baseline of a closure triangle. We can 
    test that the columns of the non-zero entry of the 3 projectors (V1, V2 and V3) correspond to a closure triangle by 
    asking the BLM which 2 apertures a baseline was constructed from (that is the column indices of non-zero entries
     in the BLM). We then ask that these index cancel each other in the closure relation. 
     
     Since we exect B = V1 * V2.conj() * V3, then the closure relation is
     
     Closure relation: (i - j) - (i - k) + (j - k)
    """
    # We expect BLM[i] = 1 and BLM[j] = -1 for i < j where i and j are non zero entries of BLM.
    # Otherwise, this test will fail
    baseline_indices, aperture_indices = np.nonzero(B.BLM)
    i = aperture_indices[::2]
    j = aperture_indices[1::2]
    assert np.array_equal(B.BLM[baseline_indices[::2], i], np.ones_like(i))
    assert np.array_equal(B.BLM[baseline_indices[1::2], j], -np.ones_like(j))
    print(f"aperture_indices = {aperture_indices}")

    baseline_1 = np.nonzero(V1)[1]
    baseline_2 = np.nonzero(V2)[1]
    baseline_3 = np.nonzero(V3)[1]

    # first baseline aperture indices
    print(f"baseline_1 = {baseline_1}")
    print(B.BLM[baseline_1])
    i1 = np.nonzero(B.BLM[baseline_1])[1][::2]
    print(f"i1 = {i1}")
    j1 = np.nonzero(B.BLM[baseline_1])[1][1::2]
    print(f"j1 = {j1}")

    # second baseline aperture indices (conjugate -> inverse sign in BLM)
    print(B.BLM[baseline_2])
    i2 = np.nonzero(B.BLM[baseline_2])[1][::2]
    k1 = np.nonzero(B.BLM[baseline_2])[1][1::2]

    # third baseline aperture indices
    print(B.BLM[baseline_3])
    j2 = np.nonzero(B.BLM[baseline_3])[1][::2]
    k2 = np.nonzero(B.BLM[baseline_3])[1][1::2]

    assert np.array_equal(i1,  i2)
    assert np.array_equal(j1, j2)
    assert np.array_equal(k1, k2)

