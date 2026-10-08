"""
Is the Ising transition pinned by Kramers-Wannier self-duality? Exact diagonalization.

For interactions that preserve KW self-duality (lam, the O'Brien-Fendley 3-spin term) and for
interactions that break it (kappa Z_j Z_{j+2}), locate the transition g_c from crossings of the
scaled odd-sector gap N*Delta_sigma(N, g) for chain lengths N and N+2 (phenomenological
renormalization), then extrapolate in 1/N^2.

Prediction (KW protection of the 'mass' operator epsilon): g_c = 1 for every lam;
g_c moves away from 1 for kappa != 0, linearly at small kappa.

Usage: python scripts/kw_protection_scan.py [--Nmax 18] [--out results/kw_scan.npz]
Output: results/kw_scan.npz (the run reported first used --Nmax 16: pairs (8,10) ... (14,16);
        --Nmax 18 adds the pair (16,18), written to results/kw_scan_N18.npz)
"""
import sys, time, argparse
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, "src")
from isingflow.kwchain import gaps

ap = argparse.ArgumentParser()
ap.add_argument("--Nmax", type=int, default=18)
ap.add_argument("--out", default="results/kw_scan.npz")
args = ap.parse_args()

CASES = [("self-dual", 0.0, 0.0), ("self-dual", 0.1, 0.0), ("self-dual", 0.2, 0.0), ("self-dual", 0.3, 0.0),
         ("KW-breaking", 0.0, -0.05), ("KW-breaking", 0.0, -0.1), ("KW-breaking", 0.0, -0.2),
         ("KW-breaking", 0.0, 0.1), ("both", 0.2, -0.1)]
Ns = list(range(8, args.Nmax - 1, 2))
res = {}
cache = {}


def scaled_gap(N, g, lam, kap):
    key = (N, round(g, 12), lam, kap)
    if key not in cache:
        cache[key] = N * gaps(N, g, lam, kap)[1]
    return cache[key]


for kind, lam, kap in CASES:
    t0 = time.time()
    gc = []
    for N in Ns:
        f = lambda g: scaled_gap(N, g, lam, kap) - scaled_gap(N + 2, g, lam, kap)
        lo, hi = 0.6, 1.6
        # bracket: first sign change on a coarse grid, skipping points deep in the ordered phase where
        # both scaled gaps are exponentially small (below 1e-6) and the sign of f is numerical noise
        # (this produced spurious "crossings" at g = 0.6 for lam = 0.3 in an earlier run).
        grid = np.linspace(lo, hi, 11)
        vals = [f(g) for g in grid]
        ok = [min(scaled_gap(N, g, lam, kap), scaled_gap(N + 2, g, lam, kap)) > 1e-6 for g in grid]
        k = next(i for i in range(10) if ok[i] and ok[i + 1] and np.sign(vals[i]) != np.sign(vals[i + 1]))
        gc.append(brentq(f, grid[k], grid[k + 1], xtol=1e-9))
    gc = np.array(gc)
    x = 1.0 / (np.array(Ns) + 1.0) ** 2
    # linear extrapolation in 1/N^2 using the three largest pairs
    p = np.polyfit(x[-3:], gc[-3:], 1)
    res[f"{kind}_lam{lam}_kap{kap}"] = dict(Ns=Ns, gc=gc, gc_inf=p[1])
    print(f"{kind:12s} lam={lam:.2f} kappa={kap:+.2f}: g_c(N,N+2) = {np.round(gc, 6)} -> extrapolated {p[1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(args.out, **{k: np.array([v["gc_inf"], *v["gc"]]) for k, v in res.items()}, Ns=np.array(Ns))
