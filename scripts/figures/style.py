"""Shared figure style for the curriculum figure scripts (Okabe-Ito, slide-readable).

Every plotting script in scripts/figures/ imports apply_style() first and uses
the OKABE_ITO colour map plus the SD_CAPTION/SEED_MARKER constants so all
figures of a run look the same.
"""

import matplotlib as mpl

# Okabe-Ito colourblind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

SD_CAPTION = "shading/bars = sample sd over 3 seeds; dots = individual seeds"
SEED_MARKER = "o"
DPI = 200


def apply_style():
    """Set rcParams for slide-readable figures."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })
