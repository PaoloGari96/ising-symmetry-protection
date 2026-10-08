"""
MCRG linearized RG analysis (Swendsen) on Wolff configurations, iterating the 2x2 majority rule.

Outputs, for each blocking level n -> n+1 and each truncation:
  * leading eigen-exponents y of the even block (thermal sector) and of the odd block
    (magnetic sector), with jackknife errors;
  * the size of the even-odd cross blocks of the full T matrix relative to their errors
    (Z2 protection in linearized form: they must vanish).

Usage: python scripts/run_mcrg.py results/cfg_nn_L256_h0.npz results/mcrg_nn_L256.npz [--nlev 4]
"""
import sys, argparse, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from isingflow.mcrg import EVEN_OPS, ODD_OPS, level_operators, mcrg_with_errors, T_matrix
from generate_wolff import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("out")
    ap.add_argument("--nlev", type=int, default=4)
    ap.add_argument("--nblocks", type=int, default=20)
    a = ap.parse_args()
    s, meta, mag = load(a.cfg)
    N = len(s)
    names = EVEN_OPS + ODD_OPS
    ne = len(EVEN_OPS)
    rng = np.random.default_rng(2024)
    t0 = time.time()
    # operators at all levels, computed in chunks to bound memory
    chunks = [level_operators(s[i:i + 500], a.nlev, names, rng) for i in range(0, N, 500)]
    ops = [np.concatenate([c[n] for c in chunks]) for n in range(a.nlev + 1)]
    print(f"operators computed ({time.time()-t0:.0f}s); N={N}, L={s.shape[-1]}", flush=True)
    out = {"names": np.array(names), "meta": str(meta)}
    for n in range(a.nlev):
        for ke in range(1, ne + 1):
            sel = list(range(ke))
            y, dy, T, _ = mcrg_with_errors(ops, n, sel, a.nblocks)
            out[f"n{n}_even{ke}_y"], out[f"n{n}_even{ke}_dy"] = y, dy
        for ko in range(1, len(ODD_OPS) + 1):
            sel = list(range(ne, ne + ko))
            y, dy, T, _ = mcrg_with_errors(ops, n, sel, a.nblocks)
            out[f"n{n}_odd{ko}_y"], out[f"n{n}_odd{ko}_dy"] = y, dy
        # full matrix: cross blocks
        y, dy, T, Tj = mcrg_with_errors(ops, n, list(range(len(names))), a.nblocks)
        nb = a.nblocks
        Terr = np.sqrt((nb - 1) / nb * ((Tj - Tj.mean(0)) ** 2).sum(0))
        cross = np.concatenate([(T[:ne, ne:] / Terr[:ne, ne:]).ravel(), (T[ne:, :ne] / Terr[ne:, :ne]).ravel()])
        out[f"n{n}_full_T"], out[f"n{n}_full_Terr"] = T, Terr
        out[f"n{n}_cross_pulls"] = cross
        # leading exponents of full matrix (mixing allowed) for comparison
        out[f"n{n}_full_y"], out[f"n{n}_full_dy"] = y, dy
        # left eigenvectors of the full T for the two relevant eigenvalues: the scaling fields
        # u = sum_b e_b K_b. Protection: u_h has no even components, u_t no odd components.
        w, vl = np.linalg.eig(T.T)
        order = np.argsort(-np.abs(w))
        for tag, idx in (("lead", order[0]), ("second", order[1])):
            e = np.real(vl[:, idx]); e = e / e[np.argmax(np.abs(e))]
            out[f"n{n}_lefteig_{tag}"] = e
            out[f"n{n}_lefteig_{tag}_y"] = np.log(np.abs(w[idx])) / np.log(2)
            print(f"   left eigvec ({tag}, y={np.log(abs(w[idx]))/np.log(2):.3f}): "
                  + " ".join(f"{nm}:{v:+.3f}" for nm, v in zip(names, e)), flush=True)
        ye = out[f"n{n}_even{ne}_y"][:2]; yo = out[f"n{n}_odd{len(ODD_OPS)}_y"][:2]
        print(f"level {n}->{n+1}: even y = {np.round(out[f'n{n}_even{ne}_y'][:3],3)} +- {np.round(out[f'n{n}_even{ne}_dy'][:3],3)} | "
              f"odd y = {np.round(out[f'n{n}_odd{len(ODD_OPS)}_y'][:2],3)} +- {np.round(out[f'n{n}_odd{len(ODD_OPS)}_dy'][:2],3)} | "
              f"cross pulls: rms={np.sqrt(np.mean(cross**2)):.2f} max={np.abs(cross).max():.2f}", flush=True)
        for ke in range(1, ne + 1):
            print(f"   even trunc {ke}: y1={out[f'n{n}_even{ke}_y'][0]:.4f}+-{out[f'n{n}_even{ke}_dy'][0]:.4f}", flush=True)
        for ko in range(1, len(ODD_OPS) + 1):
            print(f"   odd  trunc {ko}: y1={out[f'n{n}_odd{ko}_y'][0]:.4f}+-{out[f'n{n}_odd{ko}_dy'][0]:.4f}", flush=True)
    np.savez(a.out, **out)
