import xara
import numpy as np
import scipy

# statistically independant bispectrum: each B must contain 1 and only 1 baseline which is not contained in other triangles

# this gives the complete bispectrum
def full_phase_closure_operator(kpi):
    """
    Function to work with xara KPI object, computes the phase closure operator. It computes all possible triangles formed
    from the ith aperture (he aperture where phase is zero)
    BLM: Baseline Mapping: matrix of size (q, N) with +1 and -1 mapping V to a pair of aperture (kpi.BLM from xara package)
    """
    N = kpi.nbap # number of apertures in the mask
    BLM = kpi.BLM
    p = (N-1) * (N-2) // 2 # binomial coefficient (N, 3)
    print(f"There are {p} independant closure phases")
    q = kpi.nbuv # number of independant visibilities phases
    A = np.zeros((p, q)) # closure phase operator satisfying A*(V phases) = (Closure Phases)
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
            # Sanity check that this works: closure relation should always return 0 for any three objects (1,2,3) when gain is 1
            assert np.array_equal(
                np.sign(A[diag, b1]) * (np.sign(BLM[b1, i]) * 1 + np.sign(BLM[b1, j]) * 2) \
                   + np.sign(A[diag, b2]) * (np.sign(BLM[b2, j]) * 2 + np.sign(BLM[b2, k[b2k]]) * 3)\
                   + np.sign(A[diag, b3]) * (np.sign(BLM[b3, i]) * 1 + np.sign(BLM[b3, k[b3k]]) * 3),
                np.zeros(k.size)
            ), f"Closure relation is wrong!"
            A_index += k.size
    return A


def truncated_pahse_closure_operator(kpi):
    """
    Function to work with xara KPI object, computes the phase closure operator.
    BLM: Baseline Mapping: matrix of size (q, N) with +1 and -1 mapping V to a pair of aperture (kpi.BLM from xara package)
    """
    N = kpi.nbap # number of apertures in the mask
    BLM = kpi.BLM
    triangles = N * (N-1) * (N-2) // 6 # binomial coefficient (N, 3)
    print(f"There are {triangles} triangles to look at")
    p = (N-1)*(N-2)//2 # number of independant closure phases
    print(f"There are {p} independant closure phases")
    q = kpi.nbuv # number of independant visibilities phases
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


# testing
if __name__ == "__main__":
    N = 21
    circle_mask = np.zeros((N, 2))
    random_mask = 10 * np.random.normal(size=(N, 2))
    for i in range(N):
        circle_mask[i, 0] = (100 + 10 * np.random.normal()) * np.cos(2 * np.pi * i / 21)
        circle_mask[i, 1] = (100 + 10 * np.random.normal()) * np.sin(2 * np.pi * i / 21)
    kpi = xara.KPI(array=circle_mask)
    phase_closure_operator(kpi, 0)