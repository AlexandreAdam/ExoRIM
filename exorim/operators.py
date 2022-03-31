from exorim.definitions import mas2rad, pixel_grid
import numpy as np


class Operators:
    """
    parameters:
    ----------
    - mask_coordinates : coordinates (in meters) of each apertures of the mask.
    - precision : number of digits to consider when comparing uv coordinates of Fourier components.
    - redundant: Use all triangles in the mask if True. Default is to use non-redundant triangles.

    properties:
    ----------
    - nbap: number of apertures in the mask
    - VAC: Virtual Aperture Coordinates (or just mask coordinates)
    - BLM: Baseline Model. Operator that transform the aperture phase vector into visibility phases.
    - UVC: uv coordinates (in meter) -> used to compute NDFTM

    """
    def __init__(self, mask_coordinates, wavelength, redundant=False):
        self.nbap = mask_coordinates.shape[0] 
        self.VAC = mask_coordinates
        self.wavelength = wavelength

        # Build BLM matrix (mapping from aperture to baselines)
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
        self.BLM = BLM
        self.UVC = UVC
        self.nbuv = p

        # Build CPO matrix (mapping from baselines to closure triangles)
        self.CPO = self.closure_phase_operator(redundant=redundant)

    def build_operators(self, pixels, plate_scale, return_bispectrum_operator=True):
        """ Returns direct Fourier matrix operator for visibilities and bispectrum"""
        A = self.ndftm_matrix(pixels, plate_scale)
        Ainv = self.ndftm_matrix(pixels, plate_scale, inv=True)
        if return_bispectrum_operator:
            A1, A2, A3 = self.closure_fourier_matrices(A)
            return A, Ainv, A1, A2, A3
        else:
            return A, Ainv

    def ndftm_matrix(self, pixels, plate_scale, inv=False, dprec=True):
        return NDFTM(self.UVC, self.wavelength, pixels=pixels, plate_scale=plate_scale, inv=inv, dprec=dprec)

    def closure_phase_operator(self, redundant=False):
        """
        Compute the Closure Phase Operator that act on the visibility phase vector to compute
         bispectra phases (or closure phases).
        The redundancy is removed by fixing an aperture in the aperture mask when drawing all possible triangles.
        If redundant=True, then we iterate over all possible triangle
        """
        N = self.nbap
        q = (N - 1) * (N - 2) // 2 if not redundant else N * (N - 1) * (N - 2) // 6
        p = self.nbuv
        base_apertures = [0] if not redundant else list(range(N))
        CPO = np.zeros((q, p))
        CPO_index = 0
        for i in base_apertures:
            for j in range(i+1, N):
                k = np.arange(j + 1, N)
                k = np.delete(k, np.where(k == i))
                if k.size == 0:
                    break
                # find baseline indices (b1,b2,b3) from triangle vertices (i,j,k)
                b1 = np.nonzero((self.BLM[:, i] != 0) & (self.BLM[:, j] != 0))[0][0]
                b1 = np.repeat(b1, k.size)
                # b2k and b3k keep track of which k-vertice is associated with the baseline b2 and b3 respectively
                b2, b2k = np.nonzero((self.BLM[:, k] != 0) & (self.BLM[:, j] != 0)[:, np.newaxis])
                b3, b3k = np.nonzero((self.BLM[:, k] != 0) & (self.BLM[:, i] != 0)[:, np.newaxis])
                diag = np.arange(CPO_index, CPO_index + k.size)
                # signs are retrieved from Baseline Map in order to satisfy closure relation: (i - j) + (j - k) + (k - i)
                CPO[diag, b1] += self.BLM[b1, i]
                CPO[diag, b2] += self.BLM[b2, j]
                CPO[diag, b3] += self.BLM[b3, k[b3k]]
                CPO_index += k.size
        return CPO

    def closure_baseline_projectors(self):
        return closure_baseline_projectors(self.CPO)

    def closure_fourier_matrices(self, A):
        return closure_fourier_matrices(A, self.CPO)


def NDFTM(coords, wavelength, pixels, plate_scale, inv=False, dprec=True):
    '''
    (modified from F. Martinache Xara project)

    Computes the Non uniform 2D Discrete Fourier Transform Matrix to transform flattened 2D image into complex
    Fourier coefficient.

    parameters:
    ----------
    - coords : vector of baseline (u,v) coordinates where to compute the FT (usually coords=Baselines.UVC)
    - wavelength: wavelength of light observed (in meters)
    - pixels: number of pixels of mage grid (on a side)
    - plate_scale: Plate scale of the camera in mas/pixel (that is 206265/(1000 * f[mm] * pixel_density[pixel/mm])
        or simply FOV [mas] / pixels where FOV stands for field of view and f is the focal length)

    Options:
    ------
    - inv    : Boolean (default=False) : True -> computes inverse DFT matrix
    - dprec  : double precision (default=True)
    For an image of size pixels^2, the computation requires what can be a
    fairly large (N_UV x pixels^2) auxilliary matrix. Consider using pyNFFT (or equivalent)
    for Non uniform Fast Fourier Transform.

    Example:
    --------
    >> B = exorim.operators.Baselines(mask_coordinates)
    >> A = NDFTM(B.UVC, wavelength, pixels, FOV / pixels)
    '''
    m2pix = mas2rad(plate_scale) * pixels / wavelength # meter to pixels
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


def closure_baseline_projectors(CPO):
    """
    Construct projectors from visibility space to the legs of the bispectrum triangles.
    """
    bisp_i = np.where(CPO != 0)

    # selects the first non-zero entry (column wise -- baseline wise) for each closure triangle (row)
    V1_i = (bisp_i[0][0::3], bisp_i[1][0::3])
    V2_i = (bisp_i[0][1::3], bisp_i[1][1::3])
    V3_i = (bisp_i[0][2::3], bisp_i[1][2::3])

    # Projector matrices
    V1 = np.zeros(shape=CPO.shape)
    V1[V1_i] += 1.0
    V2 = np.zeros(shape=CPO.shape)
    V2[V2_i] += 1.0
    V3 = np.zeros(shape=CPO.shape)
    V3[V3_i] += 1.0
    return V1, V2, V3


def closure_fourier_matrices(A, CPO):
    """
    Project the NDFT matrix on each leg of the bispectrum triangles.
    """
    V1, V2, V3 = closure_baseline_projectors(CPO)
    A1 = V1.dot(A)
    A2 = V2.dot(A)
    A3 = V3.dot(A)
    return A1, A2, A3


def closure_phase_covariance(CPO, sigma):
    if isinstance(sigma, np.ndarray):
        assert sigma.size == CPO.shape[1], f"Baseline error vector should be of length {CPO.shape[1]}"
    sigma_operator = np.eye(CPO.shape[1]) * sigma**2
    cp_operator = CPO.dot(sigma_operator).dot(CPO.T)
    return cp_operator


# def closure_phase_covariance_inverse(CPO, sigma):
#     cp_operator = closure_phase_covariance(CPO, sigma)
#     inv = inverse(cp_operator)
#     return inv
#
#
# def closure_phase_operator_pseudo_inverse(CPO):
#     return pseudo_inverse(CPO)

