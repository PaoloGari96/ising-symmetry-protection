"""Figure: inferred odd couplings versus the sample's blocked magnetization, for all h0.

Each point = one magnetization bin of one Wolff data set. If the inferred odd couplings were a
measurement of the explicit breaking in the renormalized Hamiltonian, points with different h0
but equal m_b would separate; if they are dominated by truncation leakage, they collapse on one curve.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import plt, BLUE, ORANGE, AQUA, GREY, GREY_L, INK, INK2, title

OUT = "figures/extension"


def panel(ax, z, b, basis, k, ylabel):
    v = z[f"b{b}_{basis}_rows"]          # columns: h0, m, h, h3, dh, dh3
    h0s = sorted(set(v[:, 0]))
    shades = np.linspace(0.35, 1.0, len(h0s) - 1)
    for j, h0 in enumerate(h0s):
        sel = v[:, 0] == h0
        if h0 == 0:
            ax.errorbar(v[sel, 1], v[sel, k], v[sel, k + 2], color=GREY, marker="o", ls="none", ms=5,
                        label="h0 = 0 (both signs of m)", zorder=3)
        else:
            ax.errorbar(v[sel, 1], v[sel, k], v[sel, k + 2], color=BLUE, alpha=float(shades[j - 1]),
                        marker="s", ls="none", ms=4.5, label=f"h0 = {h0:g}", zorder=4)
    ax.axhline(0, color=INK2, lw=0.7); ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel(f"blocked magnetization of the sample, m_b (b = {b})")
    ax.set_ylabel(ylabel)


if __name__ == "__main__":
    z = np.load("results/svf_nn_L120.npz")
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
    panel(axs[0], z, 4, "phi", 2, "inferred h (corrected basis)")
    panel(axs[1], z, 4, "phi", 3, "inferred h3 (corrected basis)")
    axs[1].legend(loc="upper left", fontsize=7, ncol=1)
    axs[0].set_ylim(-0.13, 0.13)
    axs[0].text(0.02, 0.03, "points beyond |m_b| = 0.8 come from slowly\nmixing runs (h0 >= 1e-3) and carry large errors",
                transform=axs[0].transAxes, fontsize=7.2, color=INK2)
    title(axs[0], "Inferred odd couplings follow the sample's magnetization, whatever the field h0",
          "NN Ising at Kc, L = 120; six magnetization bins per data set; h0 from 0 to 3e-3")
    fig.subplots_adjust(wspace=0.32, top=0.82)
    fig.savefig(f"{OUT}/fig_state_collapse.png")
    plt.close(fig)
    print("ok")
