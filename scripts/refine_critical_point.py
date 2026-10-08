"""Refined Binder-cumulant scan near K1c(K2=0.078). Output: results/critical_point_refined.npz"""
import sys, time
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from locate_critical_point import measure, binder

rows = []
for K1 in [0.2045, 0.2055, 0.2065]:
    for L in [60, 120, 240]:
        t = time.time()
        m = measure(L, [K1, 0.078], 6000, seed=int(1e6 * K1) + L)
        U, dU = binder(m)
        rows.append((K1, L, U, dU, (m ** 2).mean()))
        print(f"K1={K1:.4f} K2=0.078 L={L:4d} U4={U:.4f}+-{dU:.4f} <m2>={(m**2).mean():.4f} ({time.time()-t:.0f}s)", flush=True)
r = np.array(rows)
np.savez("results/critical_point_refined.npz", K1=r[:, 0], L=r[:, 1], U4=r[:, 2], dU4=r[:, 3], m2=r[:, 4])
