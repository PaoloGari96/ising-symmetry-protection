"""
Blocking, features and pseudo-likelihood (inverse Ising) inference with exact Newton steps,
plus linear-response (fluctuation-dissipation) estimates of dtheta(b)/dK0.

Model for a (blocked) configuration s on an Lb x Lb periodic lattice:
    P(s_i | rest) = sigmoid(2 s_i theta . x_i),   x_i = features of the other spins.
The log pseudo-likelihood l(theta) = sum_i log sigmoid(2 s_i theta.x_i) is concave, so
Newton's method converges to the unique maximiser (no learning-rate / iteration-count issue,
unlike the 7000-step gradient descent of the original notebooks).

Feature names (all integer valued, stored as int8):
    'n1'..'n8' : shell sums n_d(i)            -> couplings K_d     (Z2-even)
    'plaq'     : 4-spin plaquette field        -> coupling K_P      (Z2-even)
    'one'      : constant 1                    -> field h           (Z2-odd)
    'phi'      : dO3/ds_i, 12 terms            -> three-spin h3     (Z2-odd, correct)
    'theta'    : 4 corner terms (thesis Eq. A.6) -> 'h3' of the thesis (Z2-odd, not a gradient)
"""
import numpy as np
from numba import njit, prange

from .lattice import shell_sum, phi_field, theta_field, plaquette_field

EVEN = {"n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "plaq", "mf"}
ODD = {"one", "phi", "theta"}


# ----------------------------------------------------------------------------------------
# Blocking
# ----------------------------------------------------------------------------------------
def block(s, b, rng):
    """Kadanoff majority rule on b x b blocks; ties (even b) broken by a fair coin.

    s: (N, L, L) int8 -> (N, L/b, L/b) int8. Identical rule to Di Carlo's Block_configuration.
    The coin is drawn independently of the configuration, so the map commutes with s -> -s
    in distribution (Z2-covariant RG map).
    """
    N, L, _ = s.shape
    assert L % b == 0
    Lb = L // b
    m = s.reshape(N, Lb, b, Lb, b).sum(axis=(2, 4), dtype=np.int16)
    out = np.sign(m).astype(np.int8)
    ties = out == 0
    if ties.any():
        out[ties] = (2 * rng.integers(0, 2, size=int(ties.sum())) - 1).astype(np.int8)
    return out


# ----------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------
def _feature_column(s16, name):
    if name.startswith("n") and name[1:].isdigit():
        return shell_sum(s16, int(name[1:]))
    if name == "mf":
        # infinite-range (Curie-Weiss) pair coupling, Z2-even: local field = magnetization of the
        # rest of the configuration, quantized as round(100 * m_{-i}) so that it fits in int8.
        L2 = s16.shape[-1] * s16.shape[-2]
        M = s16.sum(axis=(-2, -1), keepdims=True)
        return np.rint(100.0 * (M - s16) / (L2 - 1)).astype(np.int16)
    if name == "one":
        return np.ones_like(s16)
    if name == "phi":
        return phi_field(s16)
    if name == "theta":
        return theta_field(s16)
    if name == "plaq":
        return plaquette_field(s16)
    raise ValueError(name)


def features(s, basis, chunk=256):
    """s: (N, L, L) int8 -> X: (N, L*L, p) int8 with columns in the order of `basis`.

    Built in chunks of configurations to keep the memory peak close to the size of X itself."""
    N, L, _ = s.shape
    X = np.empty((N, L * L, len(basis)), dtype=np.int8)
    for c0 in range(0, N, chunk):
        s16 = s[c0:c0 + chunk].astype(np.int16)
        n = len(s16)
        for k, name in enumerate(basis):
            f = _feature_column(s16, name)
            assert np.abs(f).max() < 128
            X[c0:c0 + n, :, k] = f.reshape(n, L * L)
    return X


# ----------------------------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _site_terms(theta, x, sg, p):
    u = 0.0
    for k in range(p):
        u += theta[k] * x[k]
    z = 2.0 * sg * u
    if z >= 0.0:
        e = np.exp(-z)
        logsig = -np.log1p(e)
        smz = e / (1.0 + e)          # sigmoid(-z)
    else:
        e = np.exp(z)
        logsig = z - np.log1p(e)
        smz = 1.0 / (1.0 + e)
    return logsig, smz


@njit(cache=True, parallel=True)
def accumulate(X, S, theta, nchunks):
    """Total log-PL, gradient and observed information J = -Hessian (sums over everything)."""
    N, M, p = X.shape
    Lp = np.zeros(nchunks)
    Gp = np.zeros((nchunks, p))
    Jp = np.zeros((nchunks, p, p))
    for c in prange(nchunks):
        t0 = (N * c) // nchunks
        t1 = (N * (c + 1)) // nchunks
        xx = np.empty(p)
        for t in range(t0, t1):
            for i in range(M):
                for k in range(p):
                    xx[k] = X[t, i, k]
                sg = S[t, i]
                logsig, smz = _site_terms(theta, xx, sg, p)
                Lp[c] += logsig
                w = 2.0 * sg * smz
                cc = 4.0 * smz * (1.0 - smz)
                for k in range(p):
                    Gp[c, k] += w * xx[k]
                    for l in range(k, p):
                        Jp[c, k, l] += cc * xx[k] * xx[l]
    L_ = Lp.sum()
    G = np.zeros(p)
    J = np.zeros((p, p))
    for c in range(nchunks):
        G += Gp[c]
        J += Jp[c]
    for k in range(p):
        for l in range(k):
            J[k, l] = J[l, k]
    return L_, G, J


@njit(cache=True, parallel=True)
def per_config(X, S, theta):
    """Per-configuration gradient g_t (N,p) and information J_t (N,p,p) at theta."""
    N, M, p = X.shape
    G = np.zeros((N, p))
    J = np.zeros((N, p, p))
    for t in prange(N):
        xx = np.empty(p)
        for i in range(M):
            for k in range(p):
                xx[k] = X[t, i, k]
            sg = S[t, i]
            logsig, smz = _site_terms(theta, xx, sg, p)
            w = 2.0 * sg * smz
            cc = 4.0 * smz * (1.0 - smz)
            for k in range(p):
                G[t, k] += w * xx[k]
                for l in range(k, p):
                    J[t, k, l] += cc * xx[k] * xx[l]
        for k in range(p):
            for l in range(k):
                J[t, k, l] = J[t, l, k]
    return G, J


# ----------------------------------------------------------------------------------------
# Newton solver
# ----------------------------------------------------------------------------------------
def fit(X, S, theta0=None, tol=1e-15, maxit=60, nchunks=8, verbose=False):
    """Maximise the pseudo-likelihood by damped Newton. Returns (theta, info dict).

    Stops when the Newton decrement per site is < tol (gradient per site ~ 1e-7, parameter
    error ~ 1e-8, far below statistical errors), or when no step improves the objective at
    machine precision. The objective is concave, so the maximiser is unique."""
    p = X.shape[-1]
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=np.float64)
    nsite = X.shape[0] * X.shape[1]
    L_, G, J = accumulate(X, S, theta, nchunks)
    hist = []
    for it in range(maxit):
        step = np.linalg.solve(J, G)
        dec = float(G @ step) / nsite           # Newton decrement^2 per site
        hist.append((it, L_ / nsite, np.abs(G).max() / nsite, dec))
        if verbose:
            print(it, L_ / nsite, np.abs(G).max() / nsite, dec)
        if dec < tol:
            break
        a = 1.0
        while True:
            Ln, Gn, Jn = accumulate(X, S, theta + a * step, nchunks)
            if Ln >= L_ + 0.25 * a * float(G @ step):
                break
            a *= 0.5
            if a < 1e-6:
                break
        if a < 1e-6:                            # converged to machine precision
            break
        theta = theta + a * step
        L_, G, J = Ln, Gn, Jn
    return theta, {"loglik_per_site": L_ / nsite, "grad_max_per_site": np.abs(G).max() / nsite,
                   "newton_dec": dec, "iterations": it + 1, "J_total": J, "nsite": nsite,
                   "history": hist}


# ----------------------------------------------------------------------------------------
# Jackknife helpers
# ----------------------------------------------------------------------------------------
def jackknife_from_blocks(values_full, values_jk):
    """values_jk: (nb, ...) delete-one estimates -> (bias-corrected mean, standard error)."""
    nb = values_jk.shape[0]
    mean_jk = values_jk.mean(axis=0)
    err = np.sqrt((nb - 1) / nb * ((values_jk - mean_jk) ** 2).sum(axis=0))
    return nb * values_full - (nb - 1) * mean_jk, err


def analyse_level(X, S, Smicro=None, nblocks=20, theta0=None):
    """Fit theta at one block level, its jackknife errors, and (optionally) the linear response
    A = dtheta/dK0 = J^{-1} Cov(g, S0) with jackknife errors.

    X, S   : features/spins of the N blocked configurations (configs must be (nearly) independent
             or ordered in time so that contiguous jackknife blocks absorb autocorrelation).
    Smicro : (N, q) microscopic operator totals S0_j for the response (None -> no response).
    """
    N = X.shape[0]
    theta, info = fit(X, S, theta0=theta0)
    G, Jt = per_config(X, S, theta)               # g_t, J_t at theta*
    edges = np.linspace(0, N, nblocks + 1).astype(int)
    Gb = np.array([G[edges[k]:edges[k + 1]].sum(0) for k in range(nblocks)])
    Jb = np.array([Jt[edges[k]:edges[k + 1]].sum(0) for k in range(nblocks)])
    Jtot = Jb.sum(0)
    # one-step (linearised) delete-one estimates of theta
    th_jk = np.array([theta - np.linalg.solve(Jtot - Jb[k], Gb[k]) for k in range(nblocks)])
    th_mean, th_err = jackknife_from_blocks(theta, th_jk)
    out = {"theta": theta, "theta_err": th_err, "theta_jk": th_jk, "info": info,
           "J_mean": Jtot / N, "G": G}
    if Smicro is not None:
        Sm = np.asarray(Smicro, dtype=np.float64)
        q = Sm.shape[1]
        # sums per block for covariance with full-sample centring of S
        def response(mask_k=None):
            if mask_k is None:
                sel = slice(None)
                n = N
                Jm = Jtot / N
                Gs, Ss = G, Sm
            else:
                keep = np.ones(N, bool)
                keep[edges[mask_k]:edges[mask_k + 1]] = False
                Gs, Ss = G[keep], Sm[keep]
                n = keep.sum()
                Jm = (Jtot - Jb[mask_k]) / n
            Gc = Gs - Gs.mean(0)
            Sc = Ss - Ss.mean(0)
            C = Gc.T @ Sc / n                     # (p, q)
            return np.linalg.solve(Jm, C)
        A = response()
        A_jk = np.array([response(k) for k in range(nblocks)])
        out["A"] = A
        out["A_jk"] = A_jk
        out["A_err"] = jackknife_from_blocks(A, A_jk)[1]
    return out
