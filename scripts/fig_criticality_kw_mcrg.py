"""Figures: Binder-cumulant criticality check, KW-duality scan, MCRG exponents."""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import plt, BLUE, ORANGE, AQUA, GREY, GREY_L, INK, INK2, title

OUT = "figures/extension"
os.makedirs(OUT, exist_ok=True)
USTAR = 0.61069   # Kamieniarz & Bloete 1993; Salas & Sokal 2000


def fig_binder():
    a = np.load("results/critical_point_scan.npz")
    b = np.load("results/critical_point_refined.npz")
    rows = {}
    for m, K1, L, U, dU in zip(a["model"], a["K1"], a["L"], a["U4"], a["dU4"]):
        rows.setdefault((str(m), round(float(K1), 5)), []).append((L, U, dU))
    for K1, L, U, dU in zip(b["K1"], b["L"], b["U4"], b["dU4"]):
        rows.setdefault(("K12", round(float(K1), 5)), []).append((L, U, dU))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(USTAR, color=INK2, lw=0.8, ls="--", label="critical value U* = 0.611")
    for (m, K1), v in sorted(rows.items(), key=lambda kv: kv[0][1]):
        if m == "K12" and abs(K1 - 0.207) < 1e-9:
            continue
        v = np.array(sorted(v))
        if m == "NN":
            c, lw, z, lab = BLUE, 2.0, 5, "NN model at exact Kc = 0.4407"
        elif abs(K1 - 0.203) < 1e-9:
            c, lw, z, lab = ORANGE, 2.0, 5, "thesis point K1 = 0.203"
        elif K1 in (0.2055, 0.2065):
            c, lw, z, lab = GREY, 1.2, 3, None
        else:
            c, lw, z, lab = GREY_L, 1.0, 2, None
        ax.errorbar(v[:, 0], v[:, 1], v[:, 2], color=c, lw=lw, marker="o", ms=3.5, zorder=z, label=lab)
        if m == "K12" and lab is None:
            ax.text(v[-1, 0] * 1.06, v[-1, 1], f"{K1:g}", color=INK2, fontsize=7.5, va="center")
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([30, 60, 120, 240]); ax.set_xticklabels(["30", "60", "120", "240"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(26, 330)
    ax.set_xlabel("linear size L")
    ax.set_ylabel("Binder cumulant U4")
    ax.legend(loc="lower left")
    title(ax, "The thesis point (0.203, 0.078) is paramagnetic: U4 falls with L",
          "K2 = 0.078; curves labelled by K1. Crossing at K1c = 0.2059(2). Wolff, 4000–6000 sweeps per point")
    fig.savefig(f"{OUT}/fig_binder_criticality.png")
    plt.close(fig)


def fig_kw():
    txt = open("results/kw_scan.log").read()
    pat = re.compile(r"(\S+)\s+lam=([\d.]+) kappa=([+-][\d.]+): g_c\(N,N\+2\) = \[([^\]]+)\] -> extrapolated ([\d.]+)")
    sd, br = [], []
    for kind, lam, kap, gcs, ginf in pat.findall(txt):
        lam, kap, ginf = float(lam), float(kap), float(ginf)
        gl = np.array(gcs.split(), float)
        if kind == "self-dual":
            sd.append((lam, ginf, gl[-1]))
        elif kind == "KW-breaking":
            br.append((kap, ginf, gl[-1]))
    br.append((0.0, *[x for x in sd if x[0] == 0][0][1:]))
    sd, br = np.array(sorted(sd)), np.array(sorted(br))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.plot(br[:, 0], br[:, 1], color=ORANGE, marker="s", label="KW-breaking  κ Σ Z_j Z_{j+2}  (x = κ)")
    ax.plot(sd[:, 0], sd[:, 1], color=BLUE, marker="o", label="KW-preserving  λ Σ (XZZ + ZZX)  (x = λ)")
    p = np.polyfit(br[np.abs(br[:, 0]) <= 0.1, 0], br[np.abs(br[:, 0]) <= 0.1, 1], 1)
    ax.text(-0.155, 1.29, f"slope dg_c/dκ = {p[0]:.2f}", color=ORANGE, fontsize=8.5)
    ax.text(0.16, 1.03, f"|g_c − 1| ≤ {np.abs(sd[:,1]-1).max():.1e}", color=BLUE, fontsize=8.5)
    ax.set_xlabel("interaction strength (λ or κ)")
    ax.set_ylabel("critical transverse field g_c (N → ∞)")
    ax.legend(loc="lower left")
    title(ax, "Self-duality pins the transition at g = 1; breaking it moves g_c linearly",
          "Qubit chain H = −Σ(ZZ + gX) + interaction; exact diagonalization N = 8–16, crossings of N·Δσ")
    fig.savefig(f"{OUT}/fig_kw_duality.png")
    plt.close(fig)


def fig_mcrg(files):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(15 / 8, color=INK2, lw=0.8, ls="--"); ax.text(3.35, 15 / 8 + 0.02, "15/8", color=INK2, fontsize=8)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--"); ax.text(3.35, 1.02, "1", color=INK2, fontsize=8)
    markers = ["o", "s", "^"]
    for k, (f, lab) in enumerate(files):
        if not os.path.exists(f):
            continue
        z = np.load(f)
        nl = len([key for key in z.files if key.endswith("_even5_y")])
        n = np.arange(nl)
        yt = np.array([z[f"n{i}_even5_y"][0] for i in n]); dyt = np.array([z[f"n{i}_even5_dy"][0] for i in n])
        yh = np.array([z[f"n{i}_odd4_y"][0] for i in n]); dyh = np.array([z[f"n{i}_odd4_dy"][0] for i in n])
        off = (k - 0.5) * 0.08
        ax.errorbar(n + off, yh, dyh, color=BLUE, marker=markers[k], ls="-" if k == 0 else ":",
                    label=f"odd sector (h), {lab}")
        ax.errorbar(n + off, yt, dyt, color=ORANGE, marker=markers[k], ls="-" if k == 0 else ":",
                    label=f"even sector (ε), {lab}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["0→1", "1→2", "2→3", "3→4"])
    ax.set_xlabel("blocking step n → n+1 (2×2 majority rule)")
    ax.set_ylabel("RG eigenvalue exponent y")
    ax.set_ylim(0.8, 2.05)
    ax.legend(loc="center right", fontsize=7.5)
    title(ax, "Linearized RG: one relevant odd and one relevant even direction",
          "Leading eigenvalues of the MCRG matrix (5 even, 4 odd operators), NN model at Kc; jackknife errors")
    fig.savefig(f"{OUT}/fig_mcrg_exponents.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_binder()
    fig_kw()
    fig_mcrg([("results/mcrg_nn_L256.npz", "L = 256"), ("results/mcrg_nn_L240.npz", "L = 240"),
              ("results/mcrg_k2c_L256.npz", "K2 model at Kc, L = 256")])
    print("ok")


def fig_critical_line():
    # K2 = 0.04: Binder crossings from results/critical_line_K2_0.04_refined.log (L = 60/120: 0.3162,
    # L = 120/240: 0.3157); quoted 0.3157(5). K2 = 0.078: results/critical_point_refined.log.
    pts = np.array([[0.0, 0.5 * np.log(1 + np.sqrt(2)), 0.0], [0.04, 0.3157, 0.0005], [0.078, 0.2059, 0.0002]])
    k2 = np.linspace(0, 0.09, 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(k2, (1 - 8 * k2) / 4, color=GREY, ls="--", lw=1.4, label="mean field ('tree level'): 4K1 + 8K2 = 1")
    ax.errorbar(pts[:, 0], pts[:, 1], pts[:, 2], color=BLUE, marker="o", ms=6, lw=1.8,
                label="exact (K2 = 0, self-dual) and Binder crossings")
    ax.plot([0.078], [0.203], marker="x", color=ORANGE, ms=9, mew=2, ls="none", label="thesis point (0.203, 0.078)")
    for x, y, _ in pts:
        mf = (1 - 8 * x) / 4
        ax.annotate("", xy=(x, y), xytext=(x, mf), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
        ax.text(x + 0.002, (y + mf) / 2, f"+{y - mf:.2f}", color=INK2, fontsize=8)
    ax.annotate("thesis point: 1.4 % below K1c,\ncorrelation length \u2248 76 at L = 240", xy=(0.078, 0.203),
                xytext=(0.052, 0.31), color=ORANGE, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax.set_xlabel("next-nearest coupling K2 (Manhattan shell d = 2)")
    ax.set_ylabel("critical nearest-neighbour coupling K1c")
    ax.set_xlim(-0.005, 0.092); ax.set_ylim(0.05, 0.48)
    ax.legend(loc="upper right", fontsize=7.8)
    ax.text(0.0, 0.07, "Z2-odd direction: hc = 0 for every K2 (symmetry)", color=INK2, fontsize=8.5)
    title(ax, "The massless point moves with every even coupling; fluctuations shift it by ~0.1–0.2",
          "Critical line of the 2-shell Ising model: even relevant operator unprotected, odd one pinned at h = 0")
    fig.savefig(f"{OUT}/fig_critical_line.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_critical_line()
