"""
Reproduce the thesis protocol (notebooks/runs/h0_*) with a fast but algorithmically identical
sampler, for several independent seeds, and analyse it with
  (a) the thesis estimator: Theta basis, gradient descent, eta=0.025, 7000 steps, K init 0,
      h init = h0, h3 init 0  (emulated with the verified-identical loss/gradient), and
  (b) the exact optimum of the same loss (Newton), Theta basis, and
  (c) the exact optimum with the corrected Phi basis.

Protocol (as in the notebooks): L=240, K=(0.203, 0.078), all-up initial state, random-site
Metropolis, 300 samples 30 sweeps apart, first 100 discarded, Kadanoff majority rule
b in {2,3,4,5,6,8}.

Usage: python scripts/reproduce_thesis_protocol.py --seeds 8
Output: results/thesis_protocol.npz
"""
import sys, time, argparse
import numpy as np
sys.path.insert(0, "src")
from isingflow.mc import Sampler
from isingflow.inference import block, features, fit, accumulate

L, K = 240, [0.203, 0.078]
BS = [2, 3, 4, 5, 6, 8]
H0S = [0.0, 1e-4, 1e-3]
BASIS_T = ["n1", "n2", "n3", "n4", "one", "theta"]
BASIS_P = ["n1", "n2", "n3", "n4", "one", "phi"]


def run_chain(h0, seed, nsamp=300, sweeps=30):
    smp = Sampler(L, K, h=h0, rng_seed=seed, start="up")
    out = np.empty((nsamp, L, L), np.int8)
    for t in range(nsamp):
        smp.metropolis(sweeps)
        out[t] = smp.config()
    return out


def gd_thesis(X, S, h0, eta=0.025, nit=7000):
    """Plain gradient descent on the mean negative log-PL, exactly as in the notebooks."""
    nsite = X.shape[0] * X.shape[1]
    th = np.array([0, 0, 0, 0, h0, 0.0])
    for _ in range(nit):
        _, G, _ = accumulate(X, S, th, 4)
        th = th + eta * G / nsite           # grad of mean(-log PL) is -G/nsite
    return th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--gd_seeds", type=int, default=1, help="seeds on which to emulate the GD")
    args = ap.parse_args()
    R = args.seeds
    res = {k: np.full((len(H0S), R, len(BS), 6), np.nan) for k in ("newton_theta", "newton_phi", "gd_theta")}
    mags = np.full((len(H0S), R, 300), np.nan)
    mblk = np.full((len(H0S), R, len(BS)), np.nan)
    for ih, h0 in enumerate(H0S):
        for r in range(R):
            t0 = time.time()
            seed = 1000 * ih + r
            s = run_chain(h0, seed)
            mags[ih, r] = s.mean(axis=(1, 2))
            use = s[100:]
            rng = np.random.default_rng(seed + 7)
            for ib, b in enumerate(BS):
                sb = block(use, b, rng)
                mblk[ih, r, ib] = sb.mean()
                Sflat = sb.reshape(len(sb), -1)
                Xt = features(sb, BASIS_T)
                res["newton_theta"][ih, r, ib] = fit(Xt, Sflat)[0]
                Xp = features(sb, BASIS_P)
                res["newton_phi"][ih, r, ib] = fit(Xp, Sflat)[0]
                if r < args.gd_seeds and b in (6, 8):
                    res["gd_theta"][ih, r, ib] = gd_thesis(Xt, Sflat, h0)
            print(f"h0={h0:g} seed={r} m(last 200)={mags[ih, r, 100:].mean():.3f} "
                  f"h(b=8) newton/theta={res['newton_theta'][ih, r, -1, 4]:+.4f} "
                  f"h3(b=8)={res['newton_theta'][ih, r, -1, 5]:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
            np.savez("results/thesis_protocol.npz", H0S=H0S, BS=BS, mags=mags, mblk=mblk, **res)
