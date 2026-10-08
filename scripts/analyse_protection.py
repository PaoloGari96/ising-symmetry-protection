"""
Inverse-Ising RG flow of the odd couplings on equilibrium (Wolff) data, with jackknife errors.

For every block size b it fits, with the exact Newton optimum:
  raw    : all configurations (Z2-symmetric ensemble, both signs of m)       -> null test
  sym    : configurations + their global flips (exactly symmetric data)       -> exact zero
  plus   : only configurations with m > 0 ("pure-state" half, like a chain stuck in one sign)
and for several bases (Theta = thesis, Phi = corrected, enlarged even sectors).

Usage: python scripts/analyse_protection.py results/cfg_X.npz results/prot_X.npz [--bs 2 3 4 5 6 8]
"""
import sys, time, argparse
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.inference import block, features, analyse_level
from generate_wolff import load

BASES = {
    "thesis_theta": ["n1", "n2", "n3", "n4", "one", "theta"],
    "phi": ["n1", "n2", "n3", "n4", "one", "phi"],
    "phi_plaq": ["n1", "n2", "n3", "n4", "plaq", "one", "phi"],
    "phi_n8_plaq": ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "plaq", "one", "phi"],
}


def fit_set(sb, basis, nblocks=20):
    X = features(sb, basis)
    r = analyse_level(X, sb.reshape(len(sb), -1), nblocks=nblocks)
    return r["theta"], r["theta_err"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("out")
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--bases", nargs="+", default=list(BASES))
    ap.add_argument("--Nmax", type=int, default=None)
    a = ap.parse_args()
    s, meta, mag = load(a.cfg)
    if a.Nmax:
        s, mag = s[:a.Nmax], mag[:a.Nmax]
    print(meta, "N =", len(s), "<m> =", mag.mean(), "<|m|> =", np.abs(mag).mean(), flush=True)
    out = {"bs": np.array(a.bs), "mag": mag, "meta": str(meta)}
    rng = np.random.default_rng(12345)
    for b in a.bs:
        sb = s if b == 1 else block(s, b, rng)
        mb = sb.mean(axis=(1, 2))
        sym = np.stack([sb, -sb], axis=1).reshape(-1, *sb.shape[1:])   # each config next to its flip
        variants = {"raw": sb, "sym": sym, "plus": sb[mb > 0], "minus": sb[mb < 0]}
        for bname in a.bases:
            for vname, data in variants.items():
                if vname == "sym" and bname not in ("phi",):
                    continue
                t0 = time.time()
                th, er = fit_set(data, BASES[bname])
                out[f"b{b}_{bname}_{vname}_theta"] = th
                out[f"b{b}_{bname}_{vname}_err"] = er
                out[f"b{b}_{vname}_m"] = data.mean()
                print(f"b={b} {bname:12s} {vname:5s} N={len(data):5d} <m>={data.mean():+.3f} "
                      f"h={th[-2]:+.5f}+-{er[-2]:.5f} h3={th[-1]:+.5f}+-{er[-1]:.5f} "
                      f"K1={th[0]:.4f}+-{er[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
            np.savez(a.out, **out)
