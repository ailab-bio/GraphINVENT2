"""
The AIME Lab plot style: the palette, the matplotlib rcParams, and the two
helpers every figure in this repository needs.

Import these and set nothing else. A figure that defines its own colors, fonts
or sizes stops matching the rest of the paper, which is the whole point of
having one module:

    from metrics.plot_style import apply, add_panel_label, save_figure

    apply()
    fig, axes = plt.subplots(2, 2)
    ...
    add_panel_label(axes[0, 0], "(a)")
    save_figure(fig, results_dir, "training_curves")

Titles never go inside a plot; they belong in the figure caption. Axis labels
carry units. Dark Slate replaces black for every piece of text, axis and
annotation.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEEP_TEAL = "#2D4F54"
ROYAL_PURPLE = "#7B539E"
SOFT_LAVENDER = "#B8A0D2"
FOREST_GREEN = "#5A9448"
MUTED_PLUM = "#9E4A78"
DARK_SLATE = "#3A3D4A"

# Cycle order. Any inline SVG in results/report.html uses these same values, so
# the report and the paper figures match.
AIME_PALETTE = [
    DEEP_TEAL,
    ROYAL_PURPLE,
    SOFT_LAVENDER,
    FOREST_GREEN,
    MUTED_PLUM,
    DARK_SLATE,
]

AIME_RCPARAMS = {
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    # Axes: full bounding box, all four spines
    "axes.facecolor": "white",
    "axes.edgecolor": DARK_SLATE,
    "axes.linewidth": 1.2,
    "axes.labelcolor": DARK_SLATE,
    "axes.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.prop_cycle": plt.cycler("color", AIME_PALETTE),
    "axes.grid": False,
    # Ticks
    "xtick.color": DARK_SLATE,
    "ytick.color": DARK_SLATE,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.labelcolor": DARK_SLATE,
    # Figure and saving
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # Lines and markers
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
}


def apply() -> None:
    """Install the style. Call once, before creating any figure."""
    plt.rcParams.update(AIME_RCPARAMS)


def add_panel_label(ax: Axes, label: str) -> None:
    """
    Place a bold panel label outside the axes, left of the y-tick labels.

    Outside rather than inside, so the label never lands on the data.
    """
    ax.text(
        -0.18,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=DARK_SLATE,
    )


def save_figure(fig: Figure, results_dir: str, name: str) -> List[Path]:
    """
    Save *fig* into *results_dir* as both PNG and PDF, and return both paths.

    PNG at 300 dpi for slides and the web, PDF for LaTeX and print. *name* is a
    stem, without an extension.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = [results_dir / f"{name}.png", results_dir / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path)
    return paths
