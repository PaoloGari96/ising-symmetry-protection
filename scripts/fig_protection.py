"""Figures: (a) magnetization traces, thesis protocol vs Wolff; (b) odd couplings vs block size."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "src")
from figstyle import plt, BLUE, ORANGE, AQUA, GREY, GREY_L, INK, INK2, title
from generate_wolff import load

OUT = "figures/extension"


def fig_traces():
    tp = np.load("results/thesis_protocol.npz")
    _, _, magW = load("results/cfg_thesis_L240_h0.npz")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ih = list(tp["H0S"]).index(0.0)
    mags = tp["mags"][ih]
    t = np.arange(1, 301)
    for r in range(mags.shape[0]):
        if np.isfinite(mags[r]).all():
            ax.plot(t, mags[r], color=GREY, lw=0.9, label="thesis protocol: Metropolis from all-up, 6 seeds" if r == 0 else None)
    ax.plot(t, magW[:300], color=BLUE, lw=0.9, label="Wolff, same couplings, equilibrium")
    ax.axvspan(0, 100, color="#f1f0ec", zorder=0)
    ax.text(50, -0.62, "discarded\nin the thesis", ha="center", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlim(0, 300); ax.set_ylim(-0.75, 0.8)
    ax.set_xlabel("sample index (thesis: 30 Metropolis sweeps apart; Wolff: 2 cluster sweeps apart)")
    ax.set_ylabel("magnetization per spin m")
    ax.legend(loc="upper right")
    title(ax, "The thesis's h0 = 0 chains never leave m > 0; an ergodic chain visits both signs",
          "L = 240, K = (0.203, 0.078), h0 = 0")
    fig.savefig(f"{OUT}/fig_magnetization_traces.png")
    plt.close(fig)


def fig_odd_couplings():
    tp = np.load("results/thesis_protocol.npz")
    pr = np.load("results/prot_thesis_L240_h0.npz")
    bs_t = list(tp["BS"])
    ih = list(tp["H0S"]).index(0.0)
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True)
    for col, (basis, tkey, lab) in enumerate([("thesis_theta", "newton_theta", "thesis Θ field"),
                                              ("phi", "newton_phi", "corrected Φ field")]):
        for row, (k, nm) in enumerate([(4, "h"), (5, "h3")]):
            ax = axs[row, col]
            ax.axhline(0, color=INK2, lw=0.7)
            vals = tp[tkey][ih][:, :, k]                      # (seeds, b)
            for r in range(vals.shape[0]):
                ax.plot(bs_t, vals[r], color=GREY_L, lw=0.8, marker="o", ms=2.5,
                        label="thesis protocol, 6 seeds" if r == 0 else None)
            ax.plot(bs_t, np.nanmean(vals, 0), color=GREY, lw=1.6, label="thesis protocol, mean")
            bs, eq, eqe, plus = [], [], [], []
            for b in [2, 3, 4, 5, 6, 8]:
                key = f"b{b}_{basis}_raw_theta"
                if key in pr.files:
                    bs.append(b); eq.append(pr[key][k]); eqe.append(pr[f"b{b}_{basis}_raw_err"][k])
                    plus.append(pr[f"b{b}_{basis}_plus_theta"][k])
            if bs:
                ax.errorbar(np.array(bs) + 0.08, eq, eqe, color=BLUE, marker="o", lw=1.4,
                            label="Wolff, full ensemble, replica 1 (±1 s.e.)")
                ax.plot(bs, plus, color=ORANGE, marker="s", lw=1.2, ls="--", label="Wolff, m > 0 half only")
            if os.path.exists("results/prot_thesis_L240_h0_rep2.npz"):
                p2 = np.load("results/prot_thesis_L240_h0_rep2.npz")
                b2 = [b for b in [2, 3, 4, 5, 6, 8] if f"b{b}_{basis}_raw_theta" in p2.files]
                ax.errorbar(np.array(b2) + 0.22, [p2[f"b{b}_{basis}_raw_theta"][k] for b in b2],
                            [p2[f"b{b}_{basis}_raw_err"][k] for b in b2], color=AQUA, marker="D", ms=3.5,
                            lw=1.0, label="Wolff, full ensemble, replica 2")
            ax.set_ylabel(f"inferred {nm} ({lab})")
            if row == 1:
                ax.set_xlabel("block size b")
    axs[0, 0].legend(loc="upper left", fontsize=7.2)
    title(axs[0, 0], "At h0 = 0 the odd couplings vanish in the ergodic ensemble; the thesis offsets track the sample's sign",
          "L = 240, K = (0.203, 0.078); exact pseudo-likelihood optimum; left: thesis three-spin field Θ, right: corrected Φ")
    fig.subplots_adjust(top=0.86, hspace=0.12, wspace=0.28)
    fig.savefig(f"{OUT}/fig_odd_couplings_h0.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_traces()
    fig_odd_couplings()
    print("ok")
