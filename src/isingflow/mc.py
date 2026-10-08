"""
Monte Carlo samplers for H = -sum_d K_d S_d - h M (ferromagnetic K_d >= 0), numba-compiled.

* metropolis_sweeps : random-site single-spin-flip Metropolis, the dynamics of the original
                      code (src/IsingRG_3spin.py::Make_MontecarloStep), optionally with the
                      three-spin term (correct Phi field or the thesis' Theta field).
* wolff_steps       : Wolff single-cluster algorithm for any set of ferromagnetic pair
                      couplings; a uniform field h is included exactly through a Metropolis
                      acceptance of the cluster flip, P_acc = min(1, exp(-2 h s |C|)).

Spins live on a flat int8 array of length L*L; neighbours come from lattice.neighbor_table.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def _local_field_pairs(spins, site, nbr, kvec):
    H = 0.0
    for k in range(nbr.shape[1]):
        H += kvec[k] * spins[nbr[site, k]]
    return H


@njit(cache=True)
def _three_spin_field(spins, x, y, L, mode):
    # mode 1 = Phi (correct, 12 terms), mode 2 = Theta (thesis, 4 corner terms)
    xp = spins[((x + 1) % L) * L + y]
    xm = spins[((x - 1) % L) * L + y]
    yp = spins[x * L + (y + 1) % L]
    ym = spins[x * L + (y - 1) % L]
    corner = (xp + xm) * (yp + ym)
    if mode == 2:
        return corner
    pp = spins[((x + 1) % L) * L + (y + 1) % L]
    pm = spins[((x + 1) % L) * L + (y - 1) % L]
    mp = spins[((x - 1) % L) * L + (y + 1) % L]
    mm = spins[((x - 1) % L) * L + (y - 1) % L]
    arm_x = xp * (pp + pm) + xm * (mp + mm)
    arm_y = yp * (pp + mp) + ym * (pm + mm)
    return corner + arm_x + arm_y


@njit(cache=True)
def metropolis_sweeps(spins, L, nbr, kvec, h, h3, mode3, nsweeps):
    """nsweeps * L*L random-site Metropolis attempts. kvec[k] = coupling of neighbour column k."""
    N = L * L
    for _ in range(nsweeps * N):
        site = np.random.randint(N)
        H = _local_field_pairs(spins, site, nbr, kvec) + h
        if h3 != 0.0:
            H += h3 * _three_spin_field(spins, site // L, site % L, L, mode3)
        dE = 2.0 * H * spins[site]
        if dE <= 0.0 or np.random.random() < np.exp(-dE):
            spins[site] = -spins[site]


@njit(cache=True)
def wolff_steps(spins, nbr, padd, h, nsteps, mark, gen, cluster):
    """nsteps Wolff cluster moves. Returns (gen, total sites in proposed clusters, accepted moves).

    padd[k] = 1 - exp(-2 K_k) for neighbour column k. mark/gen avoid clearing a visited array.
    """
    N = spins.shape[0]
    nn = nbr.shape[1]
    tot = 0
    nacc = 0
    for _ in range(nsteps):
        gen += 1
        i0 = np.random.randint(N)
        s = spins[i0]
        mark[i0] = gen
        cluster[0] = i0
        csize = 1
        ptr = 0
        while ptr < csize:
            site = cluster[ptr]
            ptr += 1
            for k in range(nn):
                j = nbr[site, k]
                if mark[j] != gen and spins[j] == s:
                    if np.random.random() < padd[k]:
                        mark[j] = gen
                        cluster[csize] = j
                        csize += 1
        tot += csize
        accept = True
        if h != 0.0:
            dE = 2.0 * h * s * csize          # field energy cost of flipping the cluster
            if dE > 0.0 and np.random.random() >= np.exp(-dE):
                accept = False
        if accept:
            nacc += 1
            for q in range(csize):
                spins[cluster[q]] = -s
    return gen, tot, nacc


@njit(cache=True)
def wolff_until(spins, nbr, padd, h, target, mark, gen, cluster):
    """Wolff moves until the summed size of proposed clusters reaches `target` sites."""
    tot = 0
    nacc = 0
    nsteps = 0
    while tot < target:
        gen, t, a = wolff_steps(spins, nbr, padd, h, 1, mark, gen, cluster)
        tot += t
        nacc += a
        nsteps += 1
    return gen, tot, nacc, nsteps


class Sampler:
    """Convenience wrapper: Wolff (+ optional Metropolis sweeps) for a pair-coupling model."""

    def __init__(self, L, K, h=0.0, rng_seed=0, start="random"):
        from .lattice import neighbor_table
        self.L = L
        self.K = np.asarray(K, dtype=np.float64)
        dmax = len(self.K)
        self.nbr, self.shell, _ = neighbor_table(L, dmax)
        self.kvec = self.K[self.shell - 1].copy()
        self.padd = 1.0 - np.exp(-2.0 * self.kvec)
        self.h = float(h)
        seed(int(rng_seed))
        rs = np.random.RandomState(rng_seed)
        if start == "up":
            self.spins = np.ones(L * L, dtype=np.int8)
        else:
            self.spins = (2 * rs.randint(0, 2, L * L) - 1).astype(np.int8)
        self.mark = np.zeros(L * L, dtype=np.int64)
        self.gen = 0
        self.cluster = np.empty(L * L, dtype=np.int32)
        self.mean_cluster = None

    def wolff(self, nsteps):
        self.gen, tot, nacc = wolff_steps(self.spins, self.nbr, self.padd, self.h, nsteps,
                                          self.mark, self.gen, self.cluster)
        self.mean_cluster = tot / max(nsteps, 1)
        return nacc

    def thermalize(self, nsweeps):
        """Burn-in with a size-based stopping rule (fine for burn-in only) and calibrate the
        FIXED number of cluster moves per 'sweep' used afterwards by wolff_sweeps."""
        self.gen, tot, nacc, nsteps = wolff_until(self.spins, self.nbr, self.padd, self.h,
                                                  int(nsweeps * self.L * self.L),
                                                  self.mark, self.gen, self.cluster)
        # second pass, now in equilibrium, to calibrate <|C|>
        self.gen, tot, nacc, nsteps = wolff_until(self.spins, self.nbr, self.padd, self.h,
                                                  int(nsweeps * self.L * self.L),
                                                  self.mark, self.gen, self.cluster)
        self.steps_per_sweep = max(1.0, nsteps / nsweeps)
        self.mean_cluster = tot / max(nsteps, 1)
        return self.steps_per_sweep

    def wolff_sweeps(self, nsweeps):
        """A FIXED number of Wolff moves, ~ nsweeps * L^2 / <|C|> (calibrated by thermalize).

        NB: stopping when the summed cluster size reaches a target is a state-dependent
        stopping rule and biases the sampled distribution (checked against exact
        enumeration: <m^2> off by 8% at L=6). Never measure at such stopping times.
        """
        if not hasattr(self, "steps_per_sweep"):
            raise RuntimeError("call thermalize() first")
        n = max(1, int(round(nsweeps * self.steps_per_sweep)))
        return n, self.wolff(n)

    def metropolis(self, nsweeps, h3=0.0, mode3=1):
        metropolis_sweeps(self.spins, self.L, self.nbr, self.kvec, self.h, h3, mode3, nsweeps)

    def config(self):
        return self.spins.reshape(self.L, self.L).copy()
