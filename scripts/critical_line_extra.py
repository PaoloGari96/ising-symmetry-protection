"""Binder crossings at K2 = 0.04 to trace the critical line Kc(K2). Output: results/critical_line_K2_0.04.npz"""
import sys, time
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from locate_critical_point import measure, binder
rows = []
for K1 in [0.310, 0.318, 0.326]:
    for L in [60, 120, 240]:
        t = time.time()
        m = measure(L, [K1, 0.04], 5000, seed=int(1e6 * K1) + L + 7)
        U, dU = binder(m)
        rows.append((K1, L, U, dU))
        print(f"K1={K1:.4f} K2=0.04 L={L:4d} U4={U:.4f}+-{dU:.4f} ({time.time()-t:.0f}s)", flush=True)
r = np.array(rows)
np.savez("results/critical_line_K2_0.04.npz", K1=r[:, 0], L=r[:, 1], U4=r[:, 2], dU4=r[:, 3])
