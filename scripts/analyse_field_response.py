"""
Response of the inferred couplings to an explicit Z2-breaking field h0 (Wolff data).

For each configuration file (same L, different h0) and each block size b, fit the corrected
basis {K1..K4, h, h3(Phi)} (and optionally an enlarged even basis) with jackknife errors.
Also record the magnetization of the blocked data.

Usage: python scripts/analyse_field_response.py out.npz cfg1.npz cfg2.npz ... [--bs 2 3 4 5 6]
"""
import sys, argparse, time, json
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.inference import block, features, analyse_level
from generate_wolff import load

BASES = {"phi": ["n1", "n2", "n3", "n4", "one", "phi"],
         "big": ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "plaq", "one", "phi"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cfgs", nargs="+")
    ap.add_argument("--bs", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--bases", nargs="+", default=["phi", "big"])
    a = ap.parse_args()
    res = {"bs": np.array(a.bs), "files": np.array(a.cfgs)}
    h0s, Ls = [], []
    for ic, f in enumerate(a.cfgs):
        s, meta, mag = load(f)
        h0s.append(meta["h"]); Ls.append(meta["L"])
        res[f"c{ic}_mag"] = mag
        rng = np.random.default_rng(777 + ic)
        for b in a.bs:
            if s.shape[-1] % b:
                continue
            sb = block(s, b, rng)
            res[f"c{ic}_b{b}_m"] = sb.mean(axis=(1, 2))
            for bn in a.bases:
                t0 = time.time()
                X = features(sb, BASES[bn])
                r = analyse_level(X, sb.reshape(len(sb), -1), nblocks=20)
                res[f"c{ic}_b{b}_{bn}_theta"] = r["theta"]
                res[f"c{ic}_b{b}_{bn}_err"] = r["theta_err"]
                print(f"L={meta['L']} h0={meta['h']:g} b={b} {bn}: <m_b>={sb.mean():+.4f} h={r['theta'][-2]:+.5f}+-{r['theta_err'][-2]:.5f} "
                      f"h3={r['theta'][-1]:+.5f}+-{r['theta_err'][-1]:.5f} K1={r['theta'][0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        res["h0s"], res["Ls"] = np.array(h0s), np.array(Ls)
        np.savez(a.out, **res)
