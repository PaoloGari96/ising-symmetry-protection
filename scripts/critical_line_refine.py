"""
Refine the critical coupling at K2 = 0.04 (report sections 9.6 and 10).

critical_line_extra.py bracketed K1c(K2 = 0.04) between 0.310 and 0.318 on a coarse grid. Here two
more points inside the bracket are simulated with the same protocol (Wolff, 5000 measurements,
L = 60, 120, 240), and the crossing of U4(L) and U4(2L) is located by linear interpolation of
U4(2L) - U4(L) between the two grid points that bracket its sign change.

Usage: python scripts/critical_line_refine.py
Output: results/critical_line_K2_0.04_refined.npz and a printed table.
"""
import sys, time
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from locate_critical_point import measure, binder

old = np.load("results/critical_line_K2_0.04.npz")
rows = [tuple(r) for r in np.stack([old["K1"], old["L"], old["U4"], old["dU4"]], 1)]
for K1 in [0.316, 0.317]:
    for L in [60, 120, 240]:
        t = time.time()
        m = measure(L, [K1, 0.04], 5000, seed=int(1e6 * K1) + L + 7)
        U, dU = binder(m)
        rows.append((K1, L, U, dU))
        print(f"K1={K1:.4f} K2=0.04 L={L:4d} U4={U:.4f}+-{dU:.4f} ({time.time()-t:.0f}s)", flush=True)
r = np.array(sorted(rows))
np.savez("results/critical_line_K2_0.04_refined.npz", K1=r[:, 0], L=r[:, 1], U4=r[:, 2], dU4=r[:, 3])
K1s = np.unique(r[:, 0])
for L1, L2 in [(60, 120), (120, 240)]:
    d, e = [], []
    for K1 in K1s:
        a = r[(r[:, 0] == K1) & (r[:, 1] == L1)][0]
        b = r[(r[:, 0] == K1) & (r[:, 1] == L2)][0]
        d.append(b[2] - a[2]); e.append(np.hypot(a[3], b[3]))
    d, e = np.array(d), np.array(e)
    k = next(i for i in range(len(K1s) - 1) if d[i] < 0 <= d[i + 1])
    x0, x1, y0, y1 = K1s[k], K1s[k + 1], d[k], d[k + 1]
    kc = x0 - y0 * (x1 - x0) / (y1 - y0)
    # error: shift of the root when each difference moves by its standard error
    kc_lo = x0 - (y0 + e[k]) * (x1 - x0) / ((y1 - e[k + 1]) - (y0 + e[k]))
    kc_hi = x0 - (y0 - e[k]) * (x1 - x0) / ((y1 + e[k + 1]) - (y0 - e[k]))
    print(f"crossing L = {L1} / {L2}: U4 differences {np.round(d, 4)} at K1 = {K1s}; "
          f"K1c = {kc:.5f}  (statistical range {min(kc_lo, kc_hi):.5f} - {max(kc_lo, kc_hi):.5f})", flush=True)
