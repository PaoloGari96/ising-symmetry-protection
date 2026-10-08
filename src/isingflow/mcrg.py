"""
Swendsen's Monte Carlo renormalization group (MCRG) for the linearized RG matrix.

With an RG map sigma^(n) -> sigma^(n+1) (here: 2x2 majority rule, ties by a fair coin),

    T^(n)_{gb} = dK^(n+1)_g / dK^(n)_b ,   C(n+1,n+1) T = C(n+1,n),
    C(a,c)_{ij} = <S_i^(a) S_j^(c)> - <S_i^(a)><S_j^(c)>,

so T = C(n+1,n+1)^{-1} C(n+1,n) and its eigenvalues are 2^{y}. Operators S are lattice sums.
Z2 parity: even operators (pair products, plaquette) and odd operators (single spin, triplets,
five-spin star). Z2 symmetry of the ensemble makes C block diagonal between the two sectors,
hence T is block diagonal: the symmetry-protection statement in its linearized form.
"""
import numpy as np


def _r(s, dx, dy):
    return np.roll(np.roll(s, -dx, axis=-2), -dy, axis=-1)


EVEN_OPS = ["nn", "diag", "x2", "plaq", "knight"]
ODD_OPS = ["m", "tri", "line3", "star5"]


def operator_totals(s, names):
    """s: (N, L, L) int8 -> (N, len(names)) float64 lattice sums."""
    s = s.astype(np.int32)
    out = []
    for nm in names:
        if nm == "nn":
            v = s * (_r(s, 1, 0) + _r(s, 0, 1))
        elif nm == "diag":
            v = s * (_r(s, 1, 1) + _r(s, 1, -1))
        elif nm == "x2":
            v = s * (_r(s, 2, 0) + _r(s, 0, 2))
        elif nm == "plaq":
            v = s * _r(s, 1, 0) * _r(s, 0, 1) * _r(s, 1, 1)
        elif nm == "knight":
            v = s * (_r(s, 2, 1) + _r(s, 2, -1) + _r(s, 1, 2) + _r(s, 1, -2))
        elif nm == "m":
            v = s
        elif nm == "tri":
            v = s * (_r(s, 1, 0) + _r(s, -1, 0)) * (_r(s, 0, 1) + _r(s, 0, -1))
        elif nm == "line3":
            v = s * (_r(s, 1, 0) * _r(s, -1, 0) + _r(s, 0, 1) * _r(s, 0, -1))
        elif nm == "star5":
            v = s * _r(s, 1, 0) * _r(s, -1, 0) * _r(s, 0, 1) * _r(s, 0, -1)
        else:
            raise ValueError(nm)
        out.append(v.sum(axis=(-2, -1)).astype(np.float64))
    return np.stack(out, axis=1)


def block2(s, rng):
    """2x2 majority rule with random tie-breaking."""
    N, L, _ = s.shape
    m = s.reshape(N, L // 2, 2, L // 2, 2).sum(axis=(2, 4), dtype=np.int16)
    out = np.sign(m).astype(np.int8)
    t = out == 0
    if t.any():
        out[t] = (2 * rng.integers(0, 2, size=int(t.sum())) - 1).astype(np.int8)
    return out


def level_operators(s, nlev, names, rng):
    """Operator totals on levels 0..nlev (each level a 2x2 majority blocking of the previous)."""
    ops = [operator_totals(s, names)]
    cur = s
    for _ in range(nlev):
        cur = block2(cur, rng)
        ops.append(operator_totals(cur, names))
    return ops            # list of (N, q)


def T_matrix(Sa, Sb):
    """T = C(a,a)^{-1} C(a,b) with a = level n+1 and b = level n; Sa, Sb: (N, q)."""
    A = Sa - Sa.mean(0)
    B = Sb - Sb.mean(0)
    Caa = A.T @ A / len(A)
    Cab = A.T @ B / len(A)
    return np.linalg.solve(Caa, Cab)


def eigen_y(T, b=2.0):
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    return ev, np.log(np.abs(ev)) / np.log(b)


def mcrg_with_errors(ops, n, sel, nblocks=20, b=2.0):
    """Eigen-exponents of T^(n) restricted to operator indices `sel`, with jackknife errors.

    Returns (y_full (k,), y_err (k,), T_full, T_jk)."""
    Sa, Sb = ops[n + 1][:, sel], ops[n][:, sel]
    N = len(Sa)
    T = T_matrix(Sa, Sb)
    _, y = eigen_y(T, b)
    edges = np.linspace(0, N, nblocks + 1).astype(int)
    yj, Tj = [], []
    for k in range(nblocks):
        keep = np.ones(N, bool)
        keep[edges[k]:edges[k + 1]] = False
        Tk = T_matrix(Sa[keep], Sb[keep])
        Tj.append(Tk)
        yj.append(eigen_y(Tk, b)[1])
    yj = np.array(yj)
    err = np.sqrt((nblocks - 1) / nblocks * ((yj - yj.mean(0)) ** 2).sum(0))
    return y, err, T, np.array(Tj)
