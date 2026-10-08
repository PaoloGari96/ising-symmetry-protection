"""
Generate equilibrium configurations with the Wolff algorithm (field h via cluster acceptance)
and store them bit-packed with all metadata.

Usage example:
  python scripts/generate_wolff.py --L 240 --K 0.44068679350977147 --h 0 --N 4000 --skip 2 \
         --seed 1 --out results/cfg_nn_L240_h0.npz
"""
import sys, time, argparse, json
import numpy as np
sys.path.insert(0, "src")
from isingflow.mc import Sampler


def save(path, s, meta, mag):
    bits = np.packbits((s > 0).astype(np.uint8), axis=-1)
    np.savez_compressed(path, bits=bits, L=s.shape[-1], mag=mag, meta=json.dumps(meta))


def load(path):
    z = np.load(path)
    L = int(z["L"])
    s = np.unpackbits(z["bits"], axis=-1)[..., :L].astype(np.int8) * 2 - 1
    return s, json.loads(str(z["meta"])), z["mag"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--K", type=float, nargs="+", required=True)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--skip", type=float, default=2.0, help="Wolff 'sweeps' between saved configs")
    ap.add_argument("--therm", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    smp = Sampler(a.L, a.K, h=a.h, rng_seed=a.seed, start="random")
    sps = smp.thermalize(a.therm)
    s = np.empty((a.N, a.L, a.L), np.int8)
    mag = np.empty(a.N)
    nacc = ntot = 0
    for t in range(a.N):
        n, acc = smp.wolff_sweeps(a.skip)
        nacc += acc
        ntot += n
        s[t] = smp.config()
        mag[t] = s[t].mean()
    meta = dict(L=a.L, K=a.K, h=a.h, N=a.N, skip=a.skip, therm=a.therm, seed=a.seed,
                steps_per_sweep=sps, mean_cluster=smp.mean_cluster, acceptance=nacc / max(ntot, 1),
                algorithm="Wolff single-cluster, fixed number of moves per saved config; "
                          "field via cluster-flip acceptance min(1, exp(-2 h s |C|))",
                seconds=time.time() - t0)
    save(a.out, s, meta, mag)
    print(json.dumps(meta))
    print(f"<m>={mag.mean():+.4f} <m^2>={np.mean(mag**2):.4f} <|m|>={np.abs(mag).mean():.4f}")
