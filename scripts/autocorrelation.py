"""
Integrated autocorrelation times of the stored Wolff data sets (robustness check, report section 10).

For every results/cfg_*.npz: tau_int of the magnetization m, of |m| and of the nearest-neighbour
energy per site e = <s_i (s_{i+x} + s_{i+y})>, in units of saved configurations, with the
Madras-Sokal automatic window (smallest W with W >= c * tau_int(W), c = 6).
tau_int = 0.5 means independent samples.

Usage: python scripts/autocorrelation.py [cfg files ...]   (default: all results/cfg_*.npz)
Output: results/autocorr.npz and a printed table.
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from generate_wolff import load


def tau_int(x, c=6.0):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    if not np.any(x):
        return 0.5, 0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau, W = 0.5, 0
    for W in range(1, n // 2):
        tau = 0.5 + acf[1:W + 1].sum()
        if W >= c * tau:
            break
    return float(tau), W


def nn_energy(s, chunk=500):
    e = np.empty(len(s))
    for c0 in range(0, len(s), chunk):
        x = s[c0:c0 + chunk].astype(np.int16)
        e[c0:c0 + chunk] = (x * (np.roll(x, 1, axis=1) + np.roll(x, 1, axis=2))).mean(axis=(1, 2))
    return e


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results/cfg_*.npz"))
    out = {}
    print(f"{'data set':28s} {'N':>5s} {'h0':>7s} {'<|m|>':>6s} {'tau(m)':>7s} {'tau(|m|)':>8s} {'tau(e)':>7s}")
    for f in files:
        s, meta, mag = load(f)
        e = nn_energy(s)
        tm, _ = tau_int(mag)
        ta, _ = tau_int(np.abs(mag))
        te, _ = tau_int(e)
        name = os.path.basename(f)[4:-4]
        out[name] = np.array([meta["h"], len(mag), np.abs(mag).mean(), tm, ta, te])
        print(f"{name:28s} {len(mag):5d} {meta['h']:7.0e} {np.abs(mag).mean():6.3f} {tm:7.2f} {ta:8.2f} {te:7.2f}", flush=True)
        del s
    np.savez("results/autocorr.npz", **out)
