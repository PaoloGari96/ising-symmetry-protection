# Research log — extension "protected vs unprotected relevant operators"

Concise record of what was tested, how, and what was learned. All work was done on 2026-10-08 in one
session on a 2-core machine (Python 3.13, numpy 2.5, numba 0.68, scipy 1.18). Every run is reproducible
from `scripts/` with the seeds stored in each output's metadata; the `.log` next to each `results/*.npz`
is the table the script printed. Derivations are in `notes/theory_notes.md`. The full write-up (audit,
plan, results, interpretation, recommended thesis changes) is `reports/research_report_symmetry_protection.pdf`.

Notation. Model H = -sum_d K_d S_d - h M - h3 O3 on an L x L torus (S_d: pair sums at Manhattan distance d,
O3: L-shaped triplets). "NN at Kc" = nearest-neighbour model at Kc = ln(1+sqrt 2)/2 = 0.440687.
"Thesis point" = K = (0.203, 0.078). Blocking: b x b majority rule, ties broken by a fair coin.
Inference: pseudo-likelihood, exact Newton optimum, 20-block jackknife. Bases: {K1..K4, h, h3} with the
thesis three-spin field Theta (4 terms) or the exact local field Phi = dO3/ds (12 terms).

## 0. Validation of the new tools (`tests/test_pipeline.py`, log in `results/test_pipeline.log`)

| Test | Result |
|---|---|
| Phi = dO3/ds_i reproduces single-flip changes of O3 | 50/50 random flips exact; the thesis Theta field only 30/50 |
| Theta is not a gradient | dTheta_i/ds_j = 2 but dTheta_j/ds_i = 0 (j = i + x); Phi symmetric (2, 2) |
| Wolff and Metropolis vs exact enumeration (4x4, Kc) | <m^2>, <m^4> agree to < 0.3 % |
| Binder cumulant at exact Kc, L = 30..240 | U4 = 0.607(4), 0.613(2), 0.610(3), 0.613(2); exact 0.61069 |
| b = 1 recovery of couplings (L = 48, NN at Kc, N = 3000) | all 6 couplings within 1.4 sigma of (Kc, 0, 0, 0, 0, 0) |
| Fluctuation-dissipation identity A(b=1) = 1 (Phi basis) | diagonal 0.96-1.54 with large errors (0.15-1.0); max pull 2.4 |
| Symmetrised data: odd couplings, even/odd response blocks | exactly zero (1e-19, 1e-14) at b = 1, 2, 3 |
| Original JAX loss/gradient vs new Newton optimum | loss equal to 1e-14; original analytic gradient = JAX autodiff |
| KW chain ED: sector Hamiltonians vs full dense ED (N = 8) | identical spectra; N*Delta_sigma -> pi/2, N*Delta_eps -> 4 pi |

**Pitfall found and fixed.** A first version of the Wolff driver stopped each measurement interval when the
summed cluster size reached a target. This is a state-dependent stopping time and biases the ensemble
(<m^2> 8 % too high at L = 6, Binder 0.63 instead of 0.61 at Kc). Fixed by using a fixed number of cluster
moves per interval, calibrated once after burn-in. All results below use the fixed driver.

**Optimizer.** Newton stopping rule: decrement per site < 1e-15, or no Armijo step improves the objective at
machine precision. A first tolerance of 1e-20 stalled; 1e-15 gives gradients ~1e-7 per site, parameter
errors ~1e-8, far below statistical errors.

## 1. Criticality (`locate_critical_point.py`, `refine_critical_point.py`, `critical_line_extra.py`, `critical_line_refine.py`)

Binder cumulant along K2 = 0.078. At K1 = 0.203: U4 = 0.588, 0.558, 0.510, 0.341 for L = 30, 60, 120, 240
(decreasing: paramagnetic). Crossing between K1 = 0.2055 (U4 decreasing) and 0.2065 (increasing):
**K1c(K2 = 0.078) = 0.2059(2)**. The thesis point is 1.4 % below criticality; second-moment correlation length
at L = 240 is about 76 lattice spacings.

K2 = 0.04: first bracketed on a coarse grid (0.310, 0.318, 0.326) and quoted 0.3175(10) by interpolation;
refined with K1 = 0.316, 0.317 (L = 60, 120, 240, 5000 measurements): crossings 0.3162 (L = 60/120) and
0.3157 (L = 120/240). **K1c(K2 = 0.04) = 0.3157(5)** (supersedes 0.3175(10)).

Critical line vs mean field (4 K1 + 8 K2 = 1): fluctuation shifts +0.191 (K2 = 0), +0.146 (0.04), +0.112 (0.078);
slope dK1c/dK2 ~ -3.0 (mean field -2). Sensitivity at the thesis point:
Delta = |d ln xi / d ln K2| = nu K2 |dK1c/dK2| / (K1c - K1) ~ 80, comparable to xi/a ~ 76.

## 2. The thesis protocol, reproduced (`reproduce_thesis_protocol.py`)

Same dynamics (random-site Metropolis, all-up start, 300 samples x 30 sweeps, first 100 dropped), 6 seeds per h0.
The original JAX sampler was run for the first samples and shows the same relaxation (m = 0.64, 0.58, 0.59, 0.57).
At h0 = 0 every chain stays at m > 0 (0.17-0.44 over the kept samples).
At b = 8 (Theta basis, Newton): h = +0.003 +- 0.012, h3 = 0.017 +- 0.006 (h0 = 0); h = 0.004 +- 0.008,
h3 = 0.017 +- 0.007 (h0 = 1e-4); h = 0.047 +- 0.028, h3 = 0.029 +- 0.004 (h0 = 1e-3) (mean +- sd over seeds).
h0 = 1e-4 cannot be told apart from h0 = 0; gradient descent (7000 steps) and Newton agree within 6 %.

## 3. Odd couplings at h0 = 0 (`analyse_protection.py`; `leakage_controls.py`)

Data: thesis point L = 240, two replicas (seeds 12, 13), 4000 configurations each; NN at Kc, L = 240, 4000.
Variants per block size: raw (both signs), sym (each configuration with its flip), plus/minus halves.

* Null test (raw): rms pull of (h, h3) over b = 2-8 and both bases: 1.72 (replica 1), 0.86 (replica 2),
  1.20 (NN). Largest pull 3.6 (replica 1, b = 2, Theta h3 = -0.00037 +- 0.00010).
* Replica 1 offset: coherent at b = 2-5 (Phi: h3 ~ -2e-4); at b = 2 it sits in the second half of the chain
  (h3 = -0.00029 +- 0.00009 vs -0.00009 +- 0.00008 in the first) and comes from m < 0 configurations leaking
  more than m > 0 ones. Absent in replica 2 and in the NN data. Sampler and blocking are exactly Z2-symmetric
  in distribution, so this is a statistical fluctuation (see section 7 for error calibration).
* Symmetrised data: |theta_odd| < 3e-17 at every b (theorem T1).
* Single-sign leakage at b = 8: Theta h3 = +0.0093 (thesis point, m_b = 0.34), +0.0230 (NN, m_b = 0.69);
  equal and opposite in the m < 0 halves.
* b = 1 control (NN, L = 120, N = 3000): single-sign halves give h, h3 within 1.6 sigma of zero in both bases
  (`results/leakage_controls.log`): the leakage needs basis truncation.
* Basis enlargement (m > 0 half; net odd local field h + h3 <x3>, b = 4 / b = 8):
  NN at Kc: 0.047 / 0.070 (4 shells), 0.053 / 0.081 (+plaquette), 0.052 / 0.080 (+shells 5-8),
  0.047 / 0.075 (+infinite-range term), 0.053 / 0.086 (all). Thesis point: 0.013-0.016 / 0.022-0.029.
  No enlargement removes the leakage.

## 4. Explicit breaking: state channel vs field channel (`analyse_state_vs_field.py`, `analyse_field_response.py`)

NN at Kc, L = 60 and 120, h0 = 0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (3000 configurations each);
thesis point L = 240 with h0 = 0 (two replicas), 1e-4, 1e-3. Each data set is split into 6 bins of blocked
magnetization; WLS fit theta_odd = c m_b + R h0 + const over all bins (errors inflated by sqrt(chi2/dof)).
* L = 120: c(h3, Phi) = 0.012-0.027 per unit m_b, c(h, Phi) = -0.04 to -0.06; R not resolved
  (per unit h0: 5-48 for h, with errors as large as the values).
* L = 60 (b = 2-5): same picture. Thesis point L = 240: c resolved at every b; R(h3) reaches 3-5 sigma at
  b = 3, 6, 8 (Phi) and b = 3 (Theta), about 4-7 per unit h0. Only four data sets (two at h0 = 0) and slow
  mixing at h0 = 1e-3 (tau_int(E) = 74): suggestive of a field channel, not a measurement of it.
* In the Phi basis the inferred h is negative for positive h0 (e.g. h(b=4) = -0.037 at h0 = 1e-4, L = 120):
  the constant and three-spin features share the leakage with opposite signs.
* Caveat: with a field the Wolff update slows down (section 7); the h0 >= 3e-4 sets have few effective samples.

## 5. MCRG (`run_mcrg.py`)

Swendsen MCRG with 5 even (nn, diag, x2, plaq, knight) and 4 odd (m, tri, line3, star5) operators,
iterated 2x2 majority rule, 20-block jackknife.

| data | step | y_h | y_t | cross-block rms pull |
|---|---|---|---|---|
| NN at Kc, L = 240, N = 4000 | 0-1, 1-2, 2-3 | 1.8814, 1.8758, 1.8752(5) | 0.974(13), 1.010(13), 0.991(11) | 0.7-1.4 |
| NN at Kc, L = 256, N = 8000 | 0-1 ... 3-4 | 1.881, 1.876, 1.8746(3), 1.8748(5) | 0.954, 0.991, 1.007, 0.976 (+-0.01) | 0.98, 1.43, 0.81, 0.72 |
| K2 model at K1c = 0.2059, L = 256, N = 8000 | 0-1 ... 3-4 | 1.808, 1.845, 1.861, 1.869 | 0.996, 0.985, 0.993, 0.986 (+-0.01) | 0.84, 1.06, 0.68, 1.47 |

Truncation: y_h stable to 1e-3 with 1-4 odd operators; y_t needs >= 2 even operators (one operator: 0.90-0.97).
Scaling fields (left eigenvectors, normalised to the largest component) agree between the two models:
magnetic (m, tri, line3, star5) = (0.30, 1, 0.50, 0.20) with |even components| <= 0.003;
thermal (nn, diag, x2, plaq, knight) = (0.26, 0.35, 0.46, 0.15, 1) with |odd components| <= 0.002.
Errors are statistical only.

## 6. Kramers-Wannier bridge (`kw_protection_scan.py`)

H = -sum(ZZ + gX) + lam sum(XZZ + ZZX) + kappa sum Z_j Z_{j+2}, periodic, parity sectors. Critical field from
crossings of N*Delta_sigma for (N, N+2), linear extrapolation in 1/N^2 of the three largest pairs.

| case | N <= 16 (`kw_scan.npz`) | N <= 18 (`kw_scan_N18.npz`) |
|---|---|---|
| self-dual lam = 0 (pure TFIM) | 0.99984 | 0.99990 |
| self-dual lam = 0.1 | 1.00001 | 1.00001 |
| self-dual lam = 0.2 | 1.00045 | 1.00031 |
| self-dual lam = 0.3 | 1.00035 | 1.00037 |
| KW-breaking kappa = -0.05 | 1.08366 | 1.08374 |
| KW-breaking kappa = -0.1 | 1.16566 | 1.16575 |
| KW-breaking kappa = -0.2 | 1.32507 | 1.32517 |
| KW-breaking kappa = +0.1 | 0.82549 | 0.82550 |
| both: lam = 0.2, kappa = -0.1 | 1.13701 | 1.13686 |

KW-breaking slope dg_c/dkappa ~ -1.7. The self-dual crossings converge monotonically to 1; the linear 1/N^2
extrapolation overshoots by at most 4e-4. Bug found and fixed: the first bracket search took the first sign change
of N*Delta_sigma(N) - (N+2)*Delta_sigma(N+2) on a grid starting at g = 0.6. For lam = 0.3 both gaps there are
exponentially small and the sign is ARPACK noise, so some pairs returned g = 0.6 (in the N <= 16 run only the
(8, 10) pair, which the three-pair extrapolation excludes; in a first N <= 18 run also (10, 12) and (16, 18)).
The search now skips grid points where both scaled gaps are below 1e-6; the N <= 18 column is from the fixed code,
and for lam = 0.3 all five pairs give regular crossings (0.99355 ... 0.99844).

## 7. Robustness (`autocorrelation.py`, `error_calibration.py`)

Integrated autocorrelation times, in saved configurations (Madras-Sokal window, c = 6):

| data | tau(m) | tau(abs m) | tau(E) |
|---|---|---|---|
| all h0 = 0 sets (NN L = 60-256, thesis point, K2 model) | 0.48-0.53 | 0.55-1.12 | 1.0-1.5 |
| NN L = 120, h0 = 1e-5 / 3e-5 / 1e-4 | 0.48 / 0.55 / 0.62 | 0.99 / 1.40 / 1.70 | 1.4 / 1.8 / 3.4 |
| NN L = 120, h0 = 3e-4 / 1e-3 / 3e-3 | 5.0 / 35 / 97 | 9.0 / 35 / 97 | 32 / 197 / 316 |
| NN L = 60, h0 = 1e-3 / 3e-3 | 6.0 / 19.7 | 14.7 / 19.7 | 21.7 / 52.3 |
| thesis point L = 240, h0 = 1e-4 / 1e-3 | 2.1 / 24.3 | 2.7 / 24.3 | 7.3 / 74.3 |

Error calibration (rms pull, should be 1 if the jackknife errors are right):

| comparison | even couplings | odd couplings |
|---|---|---|
| split halves, 3 data sets x 6 b x 2 bases | 1.28 (n = 72 per basis) | 0.91 (Theta), 0.96 (Phi) (n = 36) |
| replica 1 vs replica 2, 6 b x 2 bases | 0.99 (n = 24) | 1.40 (Theta), 1.35 (Phi) (n = 12) |

Odd-coupling errors are accurate to 10-30 %; even-coupling errors are about 25 % too small.

## 8. Null test at the critical point of the thesis model (`analyse_protection.py`, follow-up queue)

K = (0.2059, 0.078), L = 256, N = 8000 (`results/prot_k2c_L256_h0.log`), b = 4 and 8, Theta and Phi bases:
raw odd pulls between 1.2 and 1.8 in magnitude (rms 1.5 over 8 correlated values); symmetrised data exactly
zero; single-sign leakage at b = 8: Theta h3 = +0.0231 for m_b = 0.67, the same c ~ 0.034 per unit m_b as
the NN model at criticality (0.0230 for m_b = 0.69).

## 9. Data provenance

Wolff data sets (all with random start, fixed number of cluster moves per saved configuration):

| file | L | K | h0 | N | sweeps between configs | burn-in sweeps | seed | cluster acceptance | seconds |
|---|---|---|---|---|---|---|---|---|---|
| `cfg_k2c_L256_h0.npz` | 256 | 0.2059, 0.078 | 0 | 8000 | 2 | 500 | 71 | 1.00 | 250 |
| `cfg_nn_L120_h0.npz` | 120 | 0.440687 | 0 | 3000 | 2 | 500 | 40 | 1.00 | 8 |
| `cfg_nn_L120_h1e-3.npz` | 120 | 0.440687 | 0.001 | 3000 | 2 | 500 | 35 | 0.32 | 9 |
| `cfg_nn_L120_h1e-4.npz` | 120 | 0.440687 | 0.0001 | 3000 | 2 | 500 | 33 | 0.61 | 8 |
| `cfg_nn_L120_h1e-5.npz` | 120 | 0.440687 | 1e-05 | 3000 | 2 | 500 | 31 | 0.95 | 8 |
| `cfg_nn_L120_h3e-3.npz` | 120 | 0.440687 | 0.003 | 3000 | 2 | 500 | 36 | 0.29 | 9 |
| `cfg_nn_L120_h3e-4.npz` | 120 | 0.440687 | 0.0003 | 3000 | 2 | 500 | 34 | 0.41 | 7 |
| `cfg_nn_L120_h3e-5.npz` | 120 | 0.440687 | 3e-05 | 3000 | 2 | 500 | 32 | 0.86 | 8 |
| `cfg_nn_L240_h0.npz` | 240 | 0.440687 | 0 | 4000 | 2 | 500 | 11 | 1.00 | 43 |
| `cfg_nn_L256_h0.npz` | 256 | 0.440687 | 0 | 8000 | 2 | 500 | 21 | 1.00 | 90 |
| `cfg_nn_L60_h0.npz` | 60 | 0.440687 | 0 | 3000 | 2 | 500 | 51 | 1.00 | 2 |
| `cfg_nn_L60_h1e-3.npz` | 60 | 0.440687 | 0.001 | 3000 | 2 | 500 | 56 | 0.36 | 2 |
| `cfg_nn_L60_h1e-4.npz` | 60 | 0.440687 | 0.0001 | 3000 | 2 | 500 | 54 | 0.86 | 2 |
| `cfg_nn_L60_h1e-5.npz` | 60 | 0.440687 | 1e-05 | 3000 | 2 | 500 | 52 | 0.99 | 2 |
| `cfg_nn_L60_h3e-3.npz` | 60 | 0.440687 | 0.003 | 3000 | 2 | 500 | 57 | 0.28 | 2 |
| `cfg_nn_L60_h3e-4.npz` | 60 | 0.440687 | 0.0003 | 3000 | 2 | 500 | 55 | 0.63 | 2 |
| `cfg_nn_L60_h3e-5.npz` | 60 | 0.440687 | 3e-05 | 3000 | 2 | 500 | 53 | 0.96 | 2 |
| `cfg_thesis_L240_h0.npz` | 240 | 0.203, 0.078 | 0 | 4000 | 2 | 500 | 12 | 1.00 | 106 |
| `cfg_thesis_L240_h0_rep2.npz` | 240 | 0.203, 0.078 | 0 | 4000 | 2 | 500 | 13 | 1.00 | 110 |
| `cfg_thesis_L240_h1e-3.npz` | 240 | 0.203, 0.078 | 0.001 | 2000 | 2 | 500 | 62 | 0.45 | 76 |
| `cfg_thesis_L240_h1e-4.npz` | 240 | 0.203, 0.078 | 0.0001 | 2000 | 2 | 500 | 61 | 0.70 | 60 |

Command for any of them: `python scripts/generate_wolff.py --L <L> --K <K...> --h <h0> --N <N> --skip 2 --seed <seed> --out results/<file>`.

Analyses and the scripts that produce them: `prot_*` analyse_protection.py; `svf_*` analyse_state_vs_field.py;
`fr_*` analyse_field_response.py; `mcrg_*` run_mcrg.py; `kw_scan*` kw_protection_scan.py; `critical_*`
locate_critical_point.py, refine_critical_point.py, critical_line_extra.py, critical_line_refine.py;
`leakage_controls` leakage_controls.py; `error_calibration` error_calibration.py; `autocorr` autocorrelation.py;
`thesis_protocol` reproduce_thesis_protocol.py. Batch order: scripts/queue_generate.sh, queue_analysis.sh,
queue_robustness.sh, queue_followup.sh. Figures: scripts/fig_*.py -> figures/extension/.
