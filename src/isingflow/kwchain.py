"""
Exact diagonalization of Ising-type qubit chains with Kramers-Wannier (KW) self-duality.

    H = - sum_j ( Z_j Z_{j+1} + g X_j )
        + lam   * sum_j ( X_j Z_{j+1} Z_{j+2} + Z_j Z_{j+1} X_{j+2} )     [KW self-dual, O'Brien-Fendley]
        + kappa * sum_j   Z_j Z_{j+2}                                     [breaks KW, keeps Z2]

Periodic boundary conditions. KW duality maps X_j <-> Z_j Z_{j+1}; for g = 1 and kappa = 0 the
model is self-dual for every lam, and the transition is pinned at g = 1 (O'Brien & Fendley,
PRL 120, 206403 (2018): Ising-critical for lam < ~0.428 in this normalization).
The Z2 parity P = prod_j X_j is used to split the Hilbert space (dimension 2^(N-1) per sector).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _bit(s, j):
    return (s >> j) & 1


def sector_hamiltonian(N, g, lam=0.0, kappa=0.0, parity=+1):
    """Sparse H in the parity sector P = +1/-1, basis |s>_P = (|s> + P|~s>)/sqrt2, s with bit 0 = 0."""
    full = (1 << N) - 1
    reps = np.arange(0, 1 << N, 2, dtype=np.int64)            # bit 0 = 0
    dim = len(reps)
    index = {}
    # map representative -> index (bit 0 = 0 states are even integers: idx = s // 2)
    rows, cols, vals = [], [], []
    zs = ((reps[:, None] >> np.arange(N)) & 1) * -2 + 1        # Z eigenvalue: bit 0 -> +1, 1 -> -1
    diag = np.zeros(dim)
    for j in range(N):
        diag -= zs[:, j] * zs[:, (j + 1) % N]
        if kappa:
            diag += kappa * zs[:, j] * zs[:, (j + 2) % N]
    rows.append(np.arange(dim)); cols.append(np.arange(dim)); vals.append(diag)

    def add_flip(j, amp):
        # amp: array (dim,) amplitude of the term X_j * (diagonal factor) acting on |s>
        t = reps ^ (1 << j)
        sign = np.ones(dim)
        flipped = (t & 1) == 1
        t = np.where(flipped, t ^ full, t)                     # back to representative
        sign = np.where(flipped, float(parity), 1.0)          # |~t> = P |t>_P component
        rows.append(t // 2); cols.append(np.arange(dim)); vals.append(amp * sign)

    for j in range(N):
        add_flip(j, -g * np.ones(dim))
        if lam:
            # X_j Z_{j+1} Z_{j+2}
            add_flip(j, lam * zs[:, (j + 1) % N] * zs[:, (j + 2) % N])
            # Z_j Z_{j+1} X_{j+2}
            add_flip((j + 2) % N, lam * zs[:, j] * zs[:, (j + 1) % N])
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return H


def low_levels(N, g, lam=0.0, kappa=0.0, k=2):
    out = {}
    for P in (+1, -1):
        H = sector_hamiltonian(N, g, lam, kappa, P)
        w = eigsh(H, k=k, which="SA", return_eigenvectors=False, tol=1e-10)
        out[P] = np.sort(w)
    return out


def gaps(N, g, lam=0.0, kappa=0.0):
    """Return (E0, Delta_sigma, Delta_eps): odd-sector gap and second even-sector gap."""
    lv = low_levels(N, g, lam, kappa)
    E0 = min(lv[+1][0], lv[-1][0])
    return E0, lv[-1][0] - lv[+1][0], lv[+1][1] - lv[+1][0]
