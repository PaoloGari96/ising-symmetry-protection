# Theory notes for the extension (to be adapted into the thesis)

## T1. Z2 protection of inferred odd couplings (any truncation)

Setting. Blocked configurations s (on a periodic lattice) are fitted with the conditional model
P_theta(s_i | rest) = sigmoid(2 s_i theta . x_i), with x_i = (x_i^even, x_i^odd). Even couplings
(K_d, plaquette, ...) have features that are ODD functions of the other spins (sums of spins);
odd couplings (h, h3, ...) have features that are EVEN functions (1, products of two spins).
The pseudo-log-likelihood per configuration is l(theta; s) = sum_i log sigmoid(2 s_i theta . x_i(s)).

Lemma. Under the global flip G: s -> -s,  l(theta_e, theta_o; G s) = l(theta_e, -theta_o; s).
Proof: s_i theta_e . x_i^e(s) is invariant (both factors flip), s_i theta_o . x_i^o(s) flips sign.

Theorem. If the data distribution (or the empirical sample) is G-invariant, the expected objective
F(theta) = E[l(theta; s)] satisfies F(theta_e, theta_o) = F(theta_e, -theta_o). F is strictly concave
(sum of log-sigmoids of linear functions, features linearly independent on the support), so its
maximiser is unique, hence theta_o* = 0 — whatever even operators are included or omitted.

Consequences.
* With symmetrised data (each configuration together with its flip) the inferred odd couplings are
  zero to machine precision (checked: 1e-19).
* The majority rule with fair-coin ties commutes with G in distribution, so an ergodic sample of a
  Z2-invariant Hamiltonian is G-invariant up to sampling noise: odd couplings are then zero up to
  O(N^{-1/2}) fluctuations.
* A non-zero odd coupling at h0 = 0 therefore requires an asymmetric sample. A flat loss direction
  only inflates the variance; it cannot bias an odd coupling (contrary to thesis App. A.6).

## T2. Single-sign leakage

In a sample confined to m > 0 (non-ergodic chain, or the m > 0 half of an ergodic one) the local
conditionals are those of the symmetric Hamiltonian (DLR equations; the global constraint on the sign
of M is irrelevant for |M| >> 1). If the basis contains the true conditional, the pseudo-likelihood is
consistent and gives theta_o = 0 even from such a sample. With a truncated basis, the best fit of the
true conditional depends on the distribution of neighbourhoods; in a magnetized sample the misfit of the
even sector has a non-zero mean, which the constant (h) and three-spin features absorb:

    theta_o(b) ~ c(b, basis) * m_b        (to leading order in m_b)

Checks: (i) at b = 1 the basis contains the microscopic Hamiltonian and both single-sign halves give
h, h3 consistent with 0; (ii) at b >= 2 the halves give equal and opposite odd couplings, |c| up to
~0.06 per unit m; (iii) the thesis protocol's offsets regress on the sample magnetization with the
same sign and size.

## T3. One-loop mass corrections with a hard Euclidean cutoff (Lambda on the loop momentum)

Fermion (QED-like, Feynman gauge):     delta m = (3 g^2 m / 16 pi^2) [ ln(Lambda^2/m^2) + O(1) ]
  -> proportional to m (chiral symmetry), only logarithmic in Lambda.

Scalar (Yukawa y phi psibar psi, fermion loop at p = 0):
  delta m_phi^2 = -(4 y^2/16 pi^2) Integral_0^{Lambda^2} dx x (x - m^2)/(x + m^2)^2
               = -(y^2/4 pi^2) [ Lambda^2 - 3 m_psi^2 ln(Lambda^2/m_psi^2) + 2 m_psi^2 + O(m^4/Lambda^2) ]
  -> additive, quadratic in Lambda and independent of m_phi. The coefficient of the logarithm,
     3 y^2 m_psi^2/(4 pi^2), matches the 1/epsilon pole of the thesis' dimensional-regularization result
     12 y^2 m_psi^2/(16 pi^2) * (2/epsilon). Dimensional regularization discards the Lambda^2 term; the
     Wilsonian/lattice picture keeps it, and it is the continuum counterpart of the O(1) (in lattice units)
     shift of the Ising critical coupling away from its mean-field value.

## T4. The goofy (r0) mass relation of the 2HDM is not protected by a positive cutoff

Potential convention: V = m11^2|P1|^2 + m22^2|P2|^2 - (m12^2 P1'P2 + h.c.) + l1/2 |P1|^4 + l2/2 |P2|^4
+ l3 |P1|^2|P2|^2 + l4 |P1'P2|^2 + [l5/2 (P1'P2)^2 + l6 |P1|^2 P1'P2 + l7 |P2|^2 P1'P2 + h.c.].
One-loop quadratic divergences (scalar part; with <phi^2> = Lambda^2/16 pi^2 per real component):

    delta m11^2 = (Lambda^2/16 pi^2) (3 l1 + 2 l3 + l4) + gauge + Yukawa(P1)
    delta m22^2 = (Lambda^2/16 pi^2) (3 l2 + 2 l3 + l4) + gauge + Yukawa(P2)

(the scalar+gauge combinations are those of the 2HDM Veltman conditions, e.g. arXiv:1709.07219 and refs.)
The goofy/r0 relations are m11^2 + m22^2 = 0, l1 = l2, l6 = -l7. Under them the two shifts are EQUAL
(the gauge parts are identical for two doublets of equal hypercharge), so

    delta(m11^2 + m22^2) = 2 (Lambda^2/16 pi^2)(3 l1 + 2 l3 + l4) + gauge + Yukawa  !=  0.

The relation survives only in mass-independent schemes (dimensional regularization), where these
tadpoles vanish. This is consistent with Trautner's remark (PLB 873 (2026) 140190) that the cutoff /
renormalization scale must flip sign under the exotic space-time transformation for the effective
potential to be goofy-invariant. Corollary: no regularization with a real positive cutoff — in particular
no lattice — can realize goofy protection of the mass relation; the protection concerns logarithmic
running and threshold structure, not power divergences.

## T5. Kramers–Wannier self-duality as protection of the even operator

Quantum chain H = -sum (Z_j Z_{j+1} + g X_j) + lam sum (X_j Z_{j+1} Z_{j+2} + Z_j Z_{j+1} X_{j+2})
+ kappa sum Z_j Z_{j+2}. KW maps X_j <-> Z_j Z_{j+1}; at g = 1 and kappa = 0 the model is self-dual for
every lam (O'Brien & Fendley, PRL 120, 206403 (2018)). The thermal deformation (g - 1) is KW-odd, so a
KW-symmetric RG flow cannot generate it: the transition stays at g = 1 for all lam below the
tricritical point, without tuning. Lattice KW is a non-invertible symmetry (Seiberg, Seifnashri, Shao,
SciPost Phys. 16, 154 (2024)): a KW-invariant chain cannot have a unique gapped ground state. In the
Majorana description KW is the one-site shift gamma_a -> gamma_{a+1} and the thermal operator is the
Majorana mass, so this "protection of a scalar mass" is chiral protection of a fermion mass in a dual
language.

Structural parallels with goofy (analogy, not evidence): both flip the sign of the mass operator, both
act non-trivially on the nearest-neighbour ("kinetic") term, neither is a symmetry of the action or
Boltzmann weight away from the protected locus, both are symmetries of the RG flow whose fixed locus is
an RG-invariant hyperplane (de Boer & Trautner, arXiv:2603.12318). Difference: KW is exact and
non-perturbative on the lattice; goofy has no positive-measure lattice realization (T4).
