"""
Controls for the single-sign leakage of inferred odd couplings (report sections 9.3 and 10).

(a) b = 1 control. Without blocking, the basis {K1..K4, h, h3} contains the true Hamiltonian
    (NN at Kc, h = h3 = 0). DLR: the m > 0 half has the same conditionals as the full ensemble,
    so even single-sign halves must give h = h3 = 0 within errors. A failure would mean that the
    leakage is not a truncation effect.
(b) Basis enlargement. Plus half of the critical NN data (L = 240) at b = 4 and 8, with the even
    sector enlarged by the plaquette, shells 5-8 and an infinite-range (Curie-Weiss) pair coupling.
    If the leakage were only the missing long-range even couplings acting as a uniform field in a
    magnetized sample, the infinite-range term would absorb it.

For every fit the "net odd local field" h + h3 <x_3> is also printed (x_3 = Theta or Phi feature,
averaged over sites and configurations): the mean push of the odd terms towards the sample's sign.

Usage: python scripts/leakage_controls.py
Output: results/leakage_controls.npz and a printed table.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.inference import block, features, analyse_level
from generate_wolff import load

E4 = ["n1", "n2", "n3", "n4"]
E8 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"]
BASES = {
    "thesis_theta": E4 + ["one", "theta"],
    "phi": E4 + ["one", "phi"],
    "phi_plaq": E4 + ["plaq", "one", "phi"],
    "phi_n8_plaq": E8 + ["plaq", "one", "phi"],
    "phi_mf": E4 + ["mf", "one", "phi"],
    "phi_n8_plaq_mf": E8 + ["plaq", "mf", "one", "phi"],
}
out = {}


def run(tag, data, bname):
    t0 = time.time()
    X = features(data, BASES[bname])
    r = analyse_level(X, data.reshape(len(data), -1), nblocks=20)
    th, er = r["theta"], r["theta_err"]
    x3 = X[:, :, -1].mean()
    net = th[-2] + th[-1] * x3
    out[f"{tag}_{bname}_theta"], out[f"{tag}_{bname}_err"] = th, er
    out[f"{tag}_{bname}_net"] = net
    print(f"{tag:22s} {bname:15s} N={len(data):5d} <m>={data.mean():+.3f} "
          f"h={th[-2]:+.5f}+-{er[-2]:.5f} (pull {th[-2]/er[-2]:+5.1f})  "
          f"h3={th[-1]:+.5f}+-{er[-1]:.5f} (pull {th[-1]/er[-1]:+5.1f})  "
          f"net odd field={net:+.5f}  K1={th[0]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    del X


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    # (a) b = 1, critical NN model, L = 120
    s, meta, mag = load("results/cfg_nn_L120_h0.npz")
    print("(a) b = 1 control:", meta["L"], meta["K"], "N =", len(s), flush=True)
    for vname, sel in (("plus", mag > 0), ("minus", mag < 0), ("raw", np.ones(len(s), bool))):
        for bname in ("thesis_theta", "phi"):
            run(f"nnL120_b1_{vname}", s[sel], bname)
    np.savez("results/leakage_controls.npz", **out)
    del s
    # (b) enlarged even bases, plus half, critical NN L = 240 and thesis point L = 240
    for cfg, lab in (("results/cfg_nn_L240_h0.npz", "nnL240"), ("results/cfg_thesis_L240_h0.npz", "thL240")):
        s, meta, mag = load(cfg)
        print(f"(b) basis enlargement: {lab}", meta["K"], "N =", len(s), flush=True)
        for b in (4, 8):
            sb = block(s, b, rng)
            plus = sb[sb.mean(axis=(1, 2)) > 0]
            for bname in ("phi", "phi_plaq", "phi_n8_plaq", "phi_mf", "phi_n8_plaq_mf"):
                run(f"{lab}_b{b}_plus", plus, bname)
            np.savez("results/leakage_controls.npz", **out)
        del s
