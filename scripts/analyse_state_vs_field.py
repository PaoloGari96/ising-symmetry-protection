"""
Separate the two channels through which inferred odd couplings respond to symmetry breaking:

  theta_odd(b) = c(b) * m_b  +  R(b) * h0  + const
                 ^ state channel           ^ Hamiltonian channel
  (truncation leakage of the sample's        (RG response to the explicit field;
   magnetization; absent for a complete       for a relevant field R ~ b^{y_h})
   basis)

Each Wolff data set (fixed L, K, h0) is split into `nbin` groups of equal size ordered by the
blocked magnetization; theta is fitted on every group (exact Newton, jackknife errors). A weighted
least-squares fit over all groups of all data sets gives c(b), R(b) with errors.

Usage: python scripts/analyse_state_vs_field.py out.npz --bs 2 3 4 5 6 cfg1.npz cfg2.npz ...
"""
import sys, argparse, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.inference import block, features, analyse_level
from generate_wolff import load

BASES = {"theta": ["n1", "n2", "n3", "n4", "one", "theta"],
         "phi": ["n1", "n2", "n3", "n4", "one", "phi"]}


def wls(y, dy, X):
    w = 1.0 / dy ** 2
    A = X.T @ (w[:, None] * X)
    coef = np.linalg.solve(A, X.T @ (w * y))
    r = y - X @ coef
    chi2 = float((w * r * r).sum())
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.inv(A) * max(chi2 / dof, 1.0)      # inflate if the scatter exceeds the errors
    return coef, np.sqrt(np.diag(cov)), chi2, dof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cfgs", nargs="+")
    ap.add_argument("--bs", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--nbin", type=int, default=6)
    ap.add_argument("--bases", nargs="+", default=["phi", "theta"])
    a = ap.parse_args()
    rows = {(b, bn): [] for b in a.bs for bn in a.bases}
    for ic, f in enumerate(a.cfgs):
        s, meta, mag = load(f)
        h0 = meta["h"]
        rng = np.random.default_rng(4242 + ic)
        for b in a.bs:
            if s.shape[-1] % b:
                continue
            sb = block(s, b, rng)
            mb = sb.mean(axis=(1, 2))
            order = np.argsort(mb)
            for grp in np.array_split(order, a.nbin):
                g = np.sort(grp)                       # keep time order inside the group
                for bn in a.bases:
                    X = features(sb[g], BASES[bn])
                    try:
                        r = analyse_level(X, sb[g].reshape(len(g), -1), nblocks=10)
                    except np.linalg.LinAlgError:      # e.g. a fully magnetized bin on a tiny lattice
                        print(f"  skipped singular bin: h0={h0:g} b={b} {bn}", flush=True)
                        continue
                    rows[(b, bn)].append((h0, mb[g].mean(), *r["theta"][-2:], *r["theta_err"][-2:]))
            print(f"{f}: h0={h0:g} b={b} done", flush=True)
    out = {}
    for (b, bn), v in rows.items():
        v = np.array(v)
        if len(v) < 4:
            continue
        out[f"b{b}_{bn}_rows"] = v
        X = np.stack([v[:, 1], v[:, 0] * 1e3, np.ones(len(v))], 1)
        for k, nm in ((2, "h"), (3, "h3")):
            coef, err, chi2, dof = wls(v[:, k], v[:, k + 2], X)
            out[f"b{b}_{bn}_{nm}_coef"], out[f"b{b}_{bn}_{nm}_err"] = coef, err
            print(f"b={b} {bn:5s} {nm:2s}: c (per unit m) = {coef[0]:+.4f}+-{err[0]:.4f} | "
                  f"R (per 1e-3 of h0) = {coef[1]:+.4f}+-{err[1]:.4f} | const {coef[2]:+.5f} | chi2/dof {chi2:.0f}/{dof}", flush=True)
    np.savez(a.out, **out)
