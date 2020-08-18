from ExoRIM.definitions import mas2rad, pixel_grid
import numpy as np
import scipy
from scipy.linalg import inv as inverse


class Baselines:
    """
    Class that holds information about the Non uniformly spaced baselines.
        nbap: Number of apertures in the mask
        VAC: Virtual Aperture Coordinates
        BLM: BaseLine Model: shape = (nbap, nbap). It is the operator that transform aperture phase vector into
            visibilities phases.
        UVC: uv coordinates (in meter) - used to compute NDFTM

    """
    def __init__(self, mask_coordinates, precision=5):
        self.nbap = mask_coordinates.shape[0]
        self.VAC = mask_coordinates
        self.precision = precision  # precision when rounding
        self._build_uv_and_model()

    def _build_uv_and_model(self):
        N = self.nbap
        mask = self.VAC
        p = N * (N-1) // 2
        UVC = np.zeros((p, 2))
        BLM = np.zeros((p, N))
        k = 0
        for i in range(N):
            for j in range(i+1, N):
                UVC[k, 0] = mask[i, 0] - mask[j, 0]
                UVC[k, 1] = mask[i, 1] - mask[j, 1]
                BLM[k, i] += 1.0
                BLM[k, j] -= 1.0
                k += 1
        # Find distinct u and v up to precision
        _, ui = np.unique(np.round(UVC[:, 0], self.precision), return_index=True)
        _, vi = np.unique(np.round(UVC[:, 1], self.precision), return_index=True)
        # We keep only the rows of the operators corresponding to the union of u_index and v_index
        distinct_baselines = np.sort(np.array(list(set(ui).union(set(vi)))))  # sort to keep row order
        self.BLM = BLM[distinct_baselines]
        self.UVC = UVC[distinct_baselines]
        self.nbuv = distinct_baselines.size
        print(f"{distinct_baselines.size} distinct baselines found. Mask has {p - distinct_baselines.size} redundant baselines")


# modified from F. Martinache Xara project
def NDFTM(coords, wavelength, pixels, plate_scale, inv=False, dprec=True):
    ''' ------------------------------------------------------------------
    Computes the (one-sided) Non uniform 2D Discrete Fourier Transform Matrix to transform flattened 2D image into complex
    Fourier coefficient. It takes as input baseline coordinate vectors (in meters) and convert them to spatial
    frequency using wavelength (in meters also).

    parameters:
    ----------
    - uv : vector of baseline (u,v) coordinates where to compute the FT
    - wavelength: wavelength of light observed (in meters)
    - pixels: number of pixels of mage grid (on a side)
    - plate_scale: Plate scale of the camera in mas/pixel (that is 206265/(1000 * f[mm] * pixel_density[pixel/mm])
        or simply FOV [mas] / pixels where FOV stands for field of view and f is the focal length)
        # Note that plate scale should be of the order of the diffraction limit for FT to give acceptable results
    Option:
    ------
    - inv    : Boolean (default=False) : True -> computes inverse DFT matrix
    - dprec  : double precision (default=True)
    For an image of size pixels^2, the computation requires what can be a
    fairly large (N_UV x pixels^2) auxilliary matrix. Consider using pyNFFT for Non uniform Fast Fourier Transform
    (tensorflow, numpy, scipy FFT implementations will not do).
    -----------------------------------
    Example of use, for an image of size isz:
    >> B = ExoRIM.operators.Baselines(mask_coordinates)
    >> FF = NDFTM(B.UVC, wavelength, pixels, FOV / pixels)
    >> FT = FF.dot(img.flatten())
    This last command returns a 1D vector FT of the img.
    ------------------------------------------------------------------ '''
    # e.g.
    # cwavel = 0.5e-6 # Wavelength [m]
    # ISZ = 128# Array size (number of pixel on a side)
    # pscale = 0.1 # plate scale [mas/pixel]
    m2pix = mas2rad(plate_scale) * pixels / wavelength
    i2pi = 1j * 2 * np.pi
    mydtype = np.complex64
    if dprec is True:
        mydtype = np.complex128
    xx, yy = pixel_grid(pixels)
    uvc = coords * m2pix  # scale correctly uv and xy so that ux + vy is in RAD
    nuv = uvc.shape[0]
    if inv is True:
        WW = np.zeros((pixels ** 2, nuv), dtype=mydtype)
        for i in range(nuv):
            # Inverse is scaled correctly with the 4 pi^2 (2D transform)
            WW[:, i] = 1./4./np.pi**2 * np.exp(i2pi * (uvc[i, 0] * xx.flatten() +
                                      uvc[i, 1] * yy.flatten()) / float(pixels))
    else:
        WW = np.zeros((nuv, pixels ** 2), dtype=mydtype)

        for i in range(nuv):
            WW[i] = np.exp(-i2pi * (uvc[i, 0] * xx.flatten() +
                                    uvc[i, 1] * yy.flatten()) / float(pixels))
    return WW


def closure_phase_covariance(CPO, sigma):
    if isinstance(sigma, np.ndarray):
        assert sigma.size == CPO.shape[1], f"Baseline error vector should be of length {CPO.shape[1]}"
    sigma_operator = np.eye(CPO.shape[1]) * sigma**2
    cp_operator = CPO.dot(sigma_operator).dot(CPO.T)
    return cp_operator


def closure_phase_covariance_inverse(CPO, sigma):
    cp_operator = closure_phase_covariance(CPO, sigma)
    inv = inverse(cp_operator)
    return inv


def closure_phase_operator(B: Baselines, fixed_aperture=0):
    """
    The phase closure operator (CPO) can act on visibilities phase vector and map them to bispectra phases. Its shape
    is (q, p):
        q: Number of independent closure phases
        p: Number of independent visibilities
    It is computed by drawing all possible triangles in the mask array from a fixed aperture (where phase is taken as
     0).
    Statistically independent bispectrum: each B must contain 1 and only 1 baseline which is not
        contained in other triangles.
    """
    # There is a bug for a redundant baseline --> test with different arrays
    N = B.nbap # number of apertures in the mask
    BLM = B.BLM
    q = (N-1) * (N-2) // 2
    print(f"There are {q} independant closure phases")
    p = B.nbuv  # number of independant visibilities phases for non-redundant mask
    A = np.zeros((q, p))  # closure phase operator satisfying A*(V phases) = (Closure Phases)
    A_index = 0  # index for A_temp
    for j in range(N): # i, j, and k select a triangle of apertures
        if fixed_aperture == j:
            continue
        # k index is vectorized
        k = np.arange(j + 1, N)
        k = np.delete(k, np.where(k == fixed_aperture))
        if k.size == 0:
            break
        # find baseline indices b1, b2 and b3 from triangle i,j,k by searching for the row index where two index were paired in Baseline Map
        b1 = np.nonzero((BLM[:, fixed_aperture] != 0) & (BLM[:, j] != 0))[0][0] # should be a single index
        b1 = np.repeat(b1, k.size) # therefore put in an array to match shape of b2 and b3
        # b2k and b3k keep track of which triangle the baseline belongs to (since indices are returned ordered by numpy nonzero)
        # in other words, the baselines b2 are associated with pairs of apertures j and k[b2k]
        b2, b2k = np.nonzero((BLM[:, k] != 0) & (BLM[:, j] != 0)[:, np.newaxis]) # index is broadcasted to shape of k
        b3, b3k = np.nonzero((BLM[:, k] != 0) & (BLM[:, fixed_aperture] != 0)[:, np.newaxis])
        diag = np.arange(A_index, A_index + k.size)
        # signs are retrieved from Baseline Map in order to satisfy closure relation: (i - j) + (j - k) + (k - i)
        A[diag, b1] += BLM[b1, fixed_aperture]
        A[diag, b2] += BLM[b2, j]
        A[diag, b3] += BLM[b3, k[b3k]]
        # Sanity check that this works: closure relation should always return 0 for any three objects (1,2,3) when gain is 1
        assert np.array_equal(
            np.sign(A[diag, b1]) * (np.sign(BLM[b1, fixed_aperture]) * 1 + np.sign(BLM[b1, j]) * 2) \
               + np.sign(A[diag, b2]) * (np.sign(BLM[b2, j]) * 2 + np.sign(BLM[b2, k[b2k]]) * 3)\
               + np.sign(A[diag, b3]) * (np.sign(BLM[b3, fixed_aperture]) * 1 + np.sign(BLM[b3, k[b3k]]) * 3),
            np.zeros(k.size)
        ), f"Closure relation is wrong!"
        A_index += k.size
    return A


def redundant_phase_closure_operator(B: Baselines):
    """
    The phase closure operator (CPO) can act on visibilities phase vector and map them to bispectra phases. Its shape
    is (q, p):
        q: Number of independent closure phases
        p: Number of independent visibilities
    It is computed by drawing all possible triangles in the mask array from a fixed aperture (where phase is taken as
     0).
    Statistically independent bispectrum: each B must contain 1 and only 1 baseline which is not
        contained in other triangles.
    """
    # There is a bug for a redundant baseline --> test with different arrays
    N = B.nbap # number of apertures in the mask
    BLM = B.BLM
    q = N * (N - 1) * (N - 2) // 6
    q_indep = (N-1) * (N-2) // 2
    print(f"There are {q_indep} independant closure phases and {q} total closure phases")
    p = B.nbuv  # number of independant visibilities phases for non-redundant mask
    A = np.zeros((q, p))  # closure phase operator satisfying A*(V phases) = (Closure Phases)
    A_index = 0  # index for A_temp
    for i in range(N):
        for j in range(i+1, N): # i, j, and k select a triangle of apertures
            # k index is vectorized
            k = np.arange(j + 1, N)
            k = np.delete(k, np.where(k == i))
            if k.size == 0:
                break
            # find baseline indices b1, b2 and b3 from triangle i,j,k by searching for the row index where two index were paired in Baseline Map
            b1 = np.nonzero((BLM[:, i] != 0) & (BLM[:, j] != 0))[0][0] # should be a single index
            b1 = np.repeat(b1, k.size) # therefore put in an array to match shape of b2 and b3
            # b2k and b3k keep track of which triangle the baseline belongs to (since indices are returned ordered by numpy nonzero)
            # in other words, the baselines b2 are associated with pairs of apertures j and k[b2k]
            b2, b2k = np.nonzero((BLM[:, k] != 0) & (BLM[:, j] != 0)[:, np.newaxis]) # index is broadcasted to shape of k
            b3, b3k = np.nonzero((BLM[:, k] != 0) & (BLM[:, i] != 0)[:, np.newaxis])
            diag = np.arange(A_index, A_index + k.size)
            # signs are retrieved from Baseline Map in order to satisfy closure relation: (i - j) + (j - k) + (k - i)
            A[diag, b1] += BLM[b1, i]
            A[diag, b2] += BLM[b2, j]
            A[diag, b3] += BLM[b3, k[b3k]]
            # Sanity check that this works: closure relation should always return 0 for any three objects (1,2,3) when gain is 1
            assert np.array_equal(
                np.sign(A[diag, b1]) * (np.sign(BLM[b1, i]) * 1 + np.sign(BLM[b1, j]) * 2) \
                   + np.sign(A[diag, b2]) * (np.sign(BLM[b2, j]) * 2 + np.sign(BLM[b2, k[b2k]]) * 3)\
                   + np.sign(A[diag, b3]) * (np.sign(BLM[b3, i]) * 1 + np.sign(BLM[b3, k[b3k]]) * 3),
                np.zeros(k.size)
            ), f"Closure relation is wrong!"
            A_index += k.size
    return A


def orthogonal_phase_closure_operator(B: Baselines):
    # kept as reference --> this method does not seem to work,
    # phase closure for unresolved source with basic noise model is not respected,
    """
    Function to work with xara KPI object, computes the phase closure operator.
    BLM: Baseline Mapping: matrix of size (q, N) with +1 and -1 mapping V to a pair of aperture (kpi.BLM from xara package)
    """
    N = B.nbap # number of apertures in the mask
    BLM = B.BLM
    triangles = N * (N-1) * (N-2) // 6 # binomial coefficient (N, 3)
    print(f"There are {triangles} triangles to look at")
    p = (N-1)*(N-2)//2 # number of independant closure phases
    print(f"There are {p} independant closure phases")
    q = B.nbuv # number of independant visibilities phases
    A = np.zeros((triangles, q)) # closure phase operator satisfying A*(V phases) = (Closure Phases)
    A_index = 0 # index for A_temp
    for i in range(N):
        for j in range(i + 1, N): # i, j, and k select a triangle of apertures
            # k index is vectorized
            k = np.arange(j + 1, N)
            if k.size == 0:
                break
            # find baseline indices b1, b2 and b3 from triangle i,j,k by searching for the row index where two index were paired in Baseline Map
            b1 = np.nonzero((BLM[:, i] != 0) & (BLM[:, j] != 0))[0][0] # should be a single index
            b1 = np.repeat(b1, k.size) # therefore put in an array to match shape of b2 and b3
            # b2k and b3k keep track of which triangle the baseline belongs to (since indices are returned ordered by numpy nonzero)
            # in other words, the baselines b2 are associated with pairs of apertures j and k[b2k]
            b2, b2k = np.nonzero((BLM[:, k] != 0) & (BLM[:, j] != 0)[:, np.newaxis]) # index is broadcasted to shape of k
            b3, b3k = np.nonzero((BLM[:, k] != 0) & (BLM[:, i] != 0)[:, np.newaxis])
            diag = np.arange(A_index, A_index + k.size)
            # signs are retrieved from Baseline Map in order to satisfy closure relation: (i - j) + (j - k) + (k - i)
            A[diag, b1] += BLM[b1, i]
            A[diag, b2] += BLM[b2, j]
            A[diag, b3] += BLM[b3, k[b3k]]
            # Sanity check that this works: closure relation should always return 0 for any three objects (1,2,3)
            assert np.array_equal(
                A[diag, b1] * (BLM[b1, i] * 1 + BLM[b1, j] * 2) \
                   + A[diag, b2] * (BLM[b2, j] * 2 + BLM[b2, k[b2k]] * 3)\
                   + A[diag, b3] * (BLM[b3, i] * 1 + BLM[b3, k[b3k]] * 3),
                np.zeros(k.size)
            ), f"Closure relation is wrong!"
            A_index += k.size
    print('Doing sparse svd')
    rank = np.linalg.matrix_rank(A.astype('double'), tol=1e-6)
    print("Closure phase operator matrix rank:", rank)
    print(f"Discards the {rank - p} smallest singular values")
    u, s, vt = scipy.sparse.linalg.svds(A.astype('double').T, k=p)
    print(f"Closure phase projection operator U shape {u.T.shape}")
    return u.T


# Debugging
if __name__ == "__main__":
    N = 21
    circle_mask = np.zeros((N, 2))
    random_mask = 10 * np.random.normal(size=(N, 2))
    for i in range(N):
        circle_mask[i, 0] = (100 + 10 * np.random.normal()) * np.cos(2 * np.pi * i / 21)
        circle_mask[i, 1] = (100 + 10 * np.random.normal()) * np.sin(2 * np.pi * i / 21)
    B = Baselines(circle_mask)
    closure_phase_operator(B, 0)
    orthogonal_phase_closure_operator(B)