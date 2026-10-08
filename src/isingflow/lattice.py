"""
Lattice geometry and operators for the extended 2D Ising model.

Conventions (identical to Di Carlo's IsingRG.py and to the thesis):

    H[s] = - sum_d K_d S_d[s] - h M[s] - h3 O3[s],     weight  exp(-H)

    S_d  = sum over UNORDERED pairs at Manhattan distance d of s_i s_j
           (shell d contains 4d sites: d=1 -> 4, d=2 -> 8, d=3 -> 12, d=4 -> 16)
    M    = sum_i s_i
    O3   = sum over "L-shaped" triplets {i, i+a, i+c}, a in {+x,-x}, c in {+y,-y}
         = sum_i s_i Theta_i ,   Theta_i = (s_{i+x}+s_{i-x}) (s_{i+y}+s_{i-y})

The local field conjugate to s_i (so that P(s_i | rest) = sigmoid(2 s_i H_i)) is

    H_i = sum_d K_d n_d(i) + h + h3 Phi_i ,   n_d(i) = sum of the 4d spins at distance d,
    Phi_i = dO3/ds_i  (12 terms: 4 with i at the corner of the triplet, 8 with i on an arm).

IMPORTANT: the thesis (Eq. A.6) and src/IsingRG_3spin.py use Theta_i (the 4 corner terms only)
as the local field of the three-spin operator. Theta is NOT the gradient of any Hamiltonian
(the mixed derivatives d Theta_i/d s_j and d Theta_j/d s_i differ), so the conditional model
with Theta is not the conditional of a Gibbs measure. Both are provided here: `phi_field`
(correct) and `theta_field` (thesis version, kept only for comparison).
"""
import numpy as np


def shell_offsets(d):
    """All (dx, dy) with |dx| + |dy| = d, in a fixed order. len = 4d."""
    out = []
    for dx in range(-d, d + 1):
        r = d - abs(dx)
        if r == 0:
            out.append((dx, 0))
        else:
            out.append((dx, r))
            out.append((dx, -r))
    return out


def neighbor_table(L, dmax):
    """Flat-index neighbour table nbr[site, k] and the shell index of each column k."""
    offs, shell = [], []
    for d in range(1, dmax + 1):
        for o in shell_offsets(d):
            offs.append(o)
            shell.append(d)
    idx = np.arange(L * L).reshape(L, L)
    nbr = np.empty((L * L, len(offs)), dtype=np.int32)
    for k, (dx, dy) in enumerate(offs):
        # spin at (x+dx, y+dy); axis 0 = x (row), axis 1 = y (column)
        nbr[:, k] = np.roll(np.roll(idx, -dx, axis=0), -dy, axis=1).ravel()
    return nbr, np.array(shell, dtype=np.int32), offs


def _roll(s, dx, dy):
    """Array whose entry at site i is s[i + (dx, dy)] (periodic). Works on (..., L, L)."""
    return np.roll(np.roll(s, -dx, axis=-2), -dy, axis=-1)


def shell_sum(s, d):
    """n_d(i): sum of the spins at Manhattan distance d from every site. s: (..., L, L)."""
    out = np.zeros(s.shape, dtype=np.int16)
    for dx, dy in shell_offsets(d):
        out += _roll(s, dx, dy)
    return out


def theta_field(s):
    """Thesis' three-spin 'field': the 4 corner products only (NOT a Hamiltonian gradient)."""
    s = s.astype(np.int16)
    return (_roll(s, 1, 0) + _roll(s, -1, 0)) * (_roll(s, 0, 1) + _roll(s, 0, -1))


def phi_field(s):
    """Correct local field of O3 = sum of L-triplets: Phi_i = dO3/ds_i (12 terms)."""
    s = s.astype(np.int16)
    xp, xm = _roll(s, 1, 0), _roll(s, -1, 0)
    yp, ym = _roll(s, 0, 1), _roll(s, 0, -1)
    corner = (xp + xm) * (yp + ym)
    # i on the horizontal arm of a triplet whose corner is i -/+ x
    arm_x = xp * (_roll(s, 1, 1) + _roll(s, 1, -1)) + xm * (_roll(s, -1, 1) + _roll(s, -1, -1))
    # i on the vertical arm of a triplet whose corner is i -/+ y
    arm_y = yp * (_roll(s, 1, 1) + _roll(s, -1, 1)) + ym * (_roll(s, 1, -1) + _roll(s, -1, -1))
    return corner + arm_x + arm_y


def plaquette_field(s):
    """Local field of the 4-spin plaquette operator P = sum_plaquettes s s s s (Z2-even)."""
    s = s.astype(np.int16)
    out = np.zeros(s.shape, dtype=np.int16)
    for ax in (1, -1):
        for ay in (1, -1):
            out += _roll(s, ax, 0) * _roll(s, 0, ay) * _roll(s, ax, ay)
    return out


def total_operators(s, dmax=4):
    """Microscopic totals per configuration: S_1..S_dmax, M, O3 (and P).

    s: (N, L, L) int8. Returns dict of (N,) float64 arrays.
    """
    s16 = s.astype(np.int16)
    out = {}
    for d in range(1, dmax + 1):
        # each unordered pair counted twice in sum_i s_i n_d(i)
        out[f"S{d}"] = 0.5 * (s16 * shell_sum(s16, d)).sum(axis=(-2, -1)).astype(np.float64)
    out["M"] = s16.sum(axis=(-2, -1)).astype(np.float64)
    out["O3"] = (s16 * theta_field(s16)).sum(axis=(-2, -1)).astype(np.float64)
    out["P"] = 0.25 * (s16 * plaquette_field(s16)).sum(axis=(-2, -1)).astype(np.float64)
    return out
