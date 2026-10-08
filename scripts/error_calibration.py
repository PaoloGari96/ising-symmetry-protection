"""
Are the jackknife errors of the inferred couplings calibrated? (report section 10)

Two independent checks on the h0 = 0 Wolff data at L = 240:
  split  : each data set is cut into its first and second halves in Monte Carlo time; both halves
           are fitted separately (exact Newton, 20-block jackknife). If the errors are right,
           pull = (theta_A - theta_B) / sqrt(err_A^2 + err_B^2) has unit variance.
  replica: the two independent replicas at the thesis point (seeds 12 and 13), full data sets,
           compared with the errors stored in results/prot_thesis_L240_h0{,_rep2}.npz.
The rms pull, separately for even (K1..K4) and odd (h, h3) couplings, estimates the factor by
which the quoted errors should be inflated.

Usage: python scripts/error_calibration.py
Output: results/error_calibration.npz and a printed table.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.inference import block, features, analyse_level
from generate_wolff import load

BASES = {"thesis_theta": ["n1", "n2", "n3", "n4", "one", "theta"],
         "phi": ["n1", "n2", "n3", "n4", "one", "phi"]}
SETS = [("results/cfg_thesis_L240_h0.npz", "thesis_rep1"),
        ("results/cfg_thesis_L240_h0_rep2.npz", "thesis_rep2"),
        ("results/cfg_nn_L240_h0.npz", "nn_Kc")]
BS = [2, 3, 4, 5, 6, 8]


def summarize(label, P):
    P = np.array(P)
    ev, od = P[:, :4].ravel(), P[:, 4:].ravel()
    print(f"{label:34s} rms pull even = {np.sqrt((ev**2).mean()):.2f} (n={ev.size}), "
          f"odd = {np.sqrt((od**2).mean()):.2f} (n={od.size}), max |odd| = {np.abs(od).max():.2f}", flush=True)
    return np.sqrt((ev ** 2).mean()), np.sqrt((od ** 2).mean())


if __name__ == "__main__":
    out = {}
    rng = np.random.default_rng(777)
    pulls = {bn: [] for bn in BASES}
    for cfg, lab in SETS:
        s, meta, mag = load(cfg)
        for b in BS:
            sb = block(s, b, rng)
            half = len(sb) // 2
            for bn, basis in BASES.items():
                t0 = time.time()
                res = []
                for part in (sb[:half], sb[half:]):
                    X = features(part, basis)
                    r = analyse_level(X, part.reshape(len(part), -1), nblocks=20)
                    res.append((r["theta"], r["theta_err"]))
                    del X
                (tA, eA), (tB, eB) = res
                p = (tA - tB) / np.sqrt(eA ** 2 + eB ** 2)
                pulls[bn].append(p)
                out[f"{lab}_b{b}_{bn}_pull"] = p
                print(f"split {lab:12s} b={b} {bn:12s} pulls K1..K4 = {np.round(p[:4], 2)}  h = {p[4]:+.2f}  h3 = {p[5]:+.2f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            np.savez("results/error_calibration.npz", **out)
        del s
    print()
    for bn in BASES:
        out[f"split_{bn}_rms"] = np.array(summarize(f"split halves, basis {bn}", pulls[bn]))
    # replica comparison with the stored full-data fits
    p1 = np.load("results/prot_thesis_L240_h0.npz")
    p2 = np.load("results/prot_thesis_L240_h0_rep2.npz")
    for bn in BASES:
        R = []
        for b in BS:
            k = f"b{b}_{bn}_raw_theta"
            if k in p1.files and k in p2.files:
                R.append((p1[k] - p2[k]) / np.sqrt(p1[f"b{b}_{bn}_raw_err"] ** 2 + p2[f"b{b}_{bn}_raw_err"] ** 2))
        out[f"replica_{bn}_rms"] = np.array(summarize(f"replica 1 vs 2, basis {bn}", R))
    np.savez("results/error_calibration.npz", **out)
