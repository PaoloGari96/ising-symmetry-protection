"""
Validation tests for the isingflow pipeline (run: python tests/test_pipeline.py).

1. Operators: Phi is the exact local field of O3; Theta (thesis) is not a gradient.
2. Samplers agree with exact enumeration (L=4) for <m^2>, <m^4>.
3. Inverse Ising at b=1 recovers the microscopic couplings (incl. h and h3 = 0).
4. Fluctuation-dissipation identity at b=1: A = dtheta/dK0 = identity (6x6) for a correct
   (Phi) basis; the Theta basis violates it in the h3 row/column.
5. Z2 null test: on symmetrised data (each configuration together with its global flip)
   every odd coupling is exactly zero and the response matrix is exactly block diagonal,
   whatever the even truncation.
6. Cross-check against the ORIGINAL code: the thesis' JAX loss/gradient (Theta basis)
   vanishes at our Newton optimum.
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from isingflow.mc import Sampler
from isingflow.lattice import total_operators, phi_field, theta_field
from isingflow.inference import block, features, fit, analyse_level

KC = 0.5 * np.log(1 + np.sqrt(2))
BASIS = ["n1", "n2", "n3", "n4", "one", "phi"]
OPS = ["S1", "S2", "S3", "S4", "M", "O3"]


def sample(L, K, n, seed, skip=2, h=0.0):
    smp = Sampler(L, K, h=h, rng_seed=seed)
    smp.thermalize(200)
    out = np.empty((n, L, L), np.int8)
    for t in range(n):
        smp.wolff_sweeps(skip)
        out[t] = smp.config()
    return out


def micro_ops(s):
    o = total_operators(s)
    return np.stack([o[k] for k in OPS], axis=1)


def test_b1_recovery_and_fdt(L=48, N=3000):
    s = sample(L, [KC], N, seed=5)
    X = features(s, BASIS)
    r = analyse_level(X, s.reshape(N, -1), Smicro=micro_ops(s), nblocks=20)
    th, dth = r["theta"], r["theta_err"]
    print("b=1 theta     :", np.round(th, 5))
    print("b=1 theta err :", np.round(dth, 5))
    target = np.array([KC, 0, 0, 0, 0, 0])
    z = (th - target) / dth
    print("pulls         :", np.round(z, 2))
    assert np.all(np.abs(z) < 4), "b=1 couplings not recovered"
    A, dA = r["A"], r["A_err"]
    print("A(b=1) diag   :", np.round(np.diag(A), 3), "+-", np.round(np.diag(dA), 3))
    off = A - np.eye(6)
    print("max |A-I|/err :", np.round(np.max(np.abs(off) / dA), 2))
    assert np.max(np.abs(off) / dA) < 5, "FDT identity violated"
    # Theta basis: the h3 direction is not the conjugate of O3
    Xt = features(s, ["n1", "n2", "n3", "n4", "one", "theta"])
    rt = analyse_level(Xt, s.reshape(N, -1), Smicro=micro_ops(s), nblocks=20)
    print("Theta basis: A[h3,O3] =", round(rt["A"][5, 5], 3), "+-", round(rt["A_err"][5, 5], 3),
          "(identity would be 1)")
    return s


def test_symmetrised_null(s):
    N = s.shape[0]
    rng = np.random.default_rng(3)
    for b in (1, 2, 3):
        sb = s if b == 1 else block(s, b, rng)
        sym = np.concatenate([sb, -sb])
        X = features(sym, BASIS)
        Smic = micro_ops(np.concatenate([s, -s]))
        r = analyse_level(X, sym.reshape(2 * N, -1), Smicro=Smic, nblocks=20)
        odd = r["theta"][4:]
        Aeo = np.abs(r["A"][:4, 4:]).max()
        Aoe = np.abs(r["A"][4:, :4]).max()
        print(f"symmetrised b={b}: odd couplings {odd}  max|A_even,odd|={Aeo:.1e} max|A_odd,even|={Aoe:.1e}")
        assert np.abs(odd).max() < 1e-10 and max(Aeo, Aoe) < 1e-8


def test_against_original_code(s):
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from IsingRG_3spin import Pseudo_Loss_fn_and_grad
    rng = np.random.default_rng(4)
    sb = block(s, 2, rng)[:400]
    X = features(sb, ["n1", "n2", "n3", "n4", "one", "theta"])
    th, info = fit(X, sb.reshape(len(sb), -1))
    loss, gK, gh, gh3 = Pseudo_Loss_fn_and_grad(jnp.array(th[:4]), jnp.array(sb, dtype=jnp.float64),
                                                float(th[4]), float(th[5]))
    g = np.concatenate([np.array(gK), [float(gh), float(gh3)]])
    print("original-code gradient at our optimum:", np.abs(g).max(), " loss", float(loss),
          " ours", -info["loglik_per_site"])
    assert np.abs(g).max() < 1e-8
    assert abs(float(loss) + info["loglik_per_site"]) < 1e-10


def test_exact_enumeration():
    L = 4
    n = L * L
    st = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).reshape(-1, L, L)
    E1 = 0.5 * (st * (np.roll(st, 1, 1) + np.roll(st, -1, 1) + np.roll(st, 1, 2) + np.roll(st, -1, 2))).sum((1, 2))
    M = st.sum((1, 2)) / n
    w = np.exp(KC * E1)
    w /= w.sum()
    ex = ((w * M ** 2).sum(), (w * M ** 4).sum())
    smp = Sampler(L, [KC], rng_seed=9)
    smp.thermalize(100)
    m = np.empty(200000)
    for t in range(len(m)):
        smp.wolff_sweeps(1)
        m[t] = smp.spins.mean()
    mc = ((m ** 2).mean(), (m ** 4).mean())
    print(f"L=4 exact <m2>,<m4> = {ex[0]:.5f},{ex[1]:.5f}   Wolff = {mc[0]:.5f},{mc[1]:.5f}")
    assert abs(ex[0] - mc[0]) < 0.004 and abs(ex[1] - mc[1]) < 0.005


if __name__ == "__main__":
    t0 = time.time()
    test_exact_enumeration()
    s = test_b1_recovery_and_fdt()
    test_symmetrised_null(s)
    test_against_original_code(s)
    print(f"ALL TESTS PASSED ({time.time() - t0:.0f}s)")
