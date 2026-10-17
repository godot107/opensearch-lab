"""Shared chart style: recessive axes, thin marks, a fixed categorical order (never cycled)."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"
INCIDENT = "#e34948"


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5,
        "lines.linewidth": 1.6, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False, "figure.dpi": 110,
    })
