"""Shared matplotlib style for the extension figures (validated palette, recessive axes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
GREY, GREY_L = "#8a8984", "#c9c8c3"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "errorbar.capsize": 2.0, "font.family": "DejaVu Sans",
})


def title(ax, text, sub=None):
    """Finding-as-title (bold) with an optional one-line subtitle, left-aligned above the axes."""
    y = 1.10 if sub else 1.03
    ax.text(0, y, text, transform=ax.transAxes, color=INK, fontweight="bold", fontsize=10, va="bottom")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
