"""
Locate the critical coupling of the thesis model (K2 = 0.078 on the Manhattan shell d = 2)
with Binder-cumulant crossings, using the Wolff algorithm.  Also checks the NN model at the
exact Kc = ln(1+sqrt 2)/2 as a control.

Usage: python scripts/locate_critical_point.py [--quick]
Output: results/critical_point_scan.npz and a printed table.
"""
import sys, time, argparse
import numpy as np
sys.path.insert(0, "src")
from isingflow.mc import Sampler

KC_NN = 0.5 * np.log(1 + np.sqrt(2))


def measure(L, K, nsweep, nskip=1, seed=0, ntherm=200):
    smp = Sampler(L, K, rng_seed=seed)
    smp.thermalize(ntherm)
    m = np.empty(nsweep)
    for t in range(nsweep):
        smp.wolff_sweeps(nskip)
        m[t] = smp.spins.mean()
    return m


def binder(m, nb=20):
    m2, m4 = m ** 2, m ** 4
    U = lambda a, b: 1 - b.mean() / (3 * a.mean() ** 2)
    full = U(m2, m4)
    idx = np.array_split(np.arange(len(m)), nb)
    jk = np.array([U(np.delete(m2, i), np.delete(m4, i)) for i in idx])
    err = np.sqrt((nb - 1) / nb * ((jk - jk.mean()) ** 2).sum())
    return full, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    Ls = [30, 60, 120, 240]
    nsw = 1000 if args.quick else 4000
    rows = []
    # control: NN model at exact Kc
    for L in Ls:
        t = time.time()
        m = measure(L, [KC_NN], nsw, seed=100 + L)
        U, dU = binder(m)
        rows.append(("NN", KC_NN, 0.0, L, U, dU, (m ** 2).mean()))
        print(f"NN   K1={KC_NN:.5f} K2=0     L={L:4d} U4={U:.4f}+-{dU:.4f} <m2>={(m**2).mean():.4f}  ({time.time()-t:.0f}s)", flush=True)
    for K1 in [0.195, 0.199, 0.203, 0.207, 0.211]:
        for L in Ls:
            t = time.time()
            m = measure(L, [K1, 0.078], nsw, seed=int(1e5 * K1) + L)
            U, dU = binder(m)
            rows.append(("K12", K1, 0.078, L, U, dU, (m ** 2).mean()))
            print(f"K12  K1={K1:.3f}   K2=0.078 L={L:4d} U4={U:.4f}+-{dU:.4f} <m2>={(m**2).mean():.4f}  ({time.time()-t:.0f}s)", flush=True)
    np.savez("results/critical_point_scan.npz",
             model=np.array([r[0] for r in rows]), K1=np.array([r[1] for r in rows]),
             K2=np.array([r[2] for r in rows]), L=np.array([r[3] for r in rows]),
             U4=np.array([r[4] for r in rows]), dU4=np.array([r[5] for r in rows]),
             m2=np.array([r[6] for r in rows]), nsweep=nsw)
