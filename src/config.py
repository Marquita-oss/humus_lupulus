"""
config.py – Global constants, colour palettes, and matplotlib/seaborn style
for the Humulus lupulus cytotoxicity analysis pipeline.

All plotting constants are defined here so every module shares a single
source of truth.  Colours follow the Paul Tol Vibrant palette for
categorical data and the RdBu / crest colormaps for continuous data.
"""

import os
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths  (relative to the repository root)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Cell-line metadata
# ---------------------------------------------------------------------------
ORDEN_LINEAS = ["MCF-7", "MDA", "AGS", "GES-1"]
LINEAS_TUMORALES = ["MCF-7", "MDA", "AGS"]
ACTIVOS = ["AT", "T3", "PN", "F1", "F2", "F3", "F4", "F5", "F7", "F8"]

# ---------------------------------------------------------------------------
# Colour system  (Paul Tol Vibrant – colour-blind safe)
# ---------------------------------------------------------------------------
COLOR_LINEA = {
    "MCF-7": "#0077BB",   # vibrant blue  (tumour)
    "MDA":   "#009988",   # vibrant teal  (tumour)
    "AGS":   "#EE7733",   # vibrant orange (tumour)
    "GES-1": "#EE3377",   # vibrant magenta (healthy, reference)
}
COLOR_ALARMA = "#C1272D"  # crimson – reserved for thresholds & alarm

# Binary grouping (dendrogram, LDA)
COLOR_CITOTOXICO = "#C1272D"    # cytotoxic group
COLOR_BAJA_CITOT = "#28417a"    # low-cytotoxicity group

# ML-model palette  (crest muted gradient)
PALETA_MODELOS = {
    "Decision tree":       "#71b490",
    "Random Forest":       "#3b8b8d",
    "SVM (RBF)":           "#5e7ca5",
    "Logistic regression": "#28417a",
    "MLP":                 "#8a5fa8",
}
ESTILO_MODELOS = {
    "Decision tree":       "--",
    "Random Forest":       "-",
    "SVM (RBF)":           ":",
    "Logistic regression": "-.",
    "MLP":                 (0, (1, 1)),
}

# Viability divergent colourmap  (RdBu centred at 100 %)
CMAP_VIAB = plt.get_cmap("RdBu")

def norma_viab(vmin=0, vcenter=100, vmax=175):
    """Two-slope normalisation centred at 100 % (solvent control)."""
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

def color_viab(v, norm=None):
    """Return the divergent colour for a given viability value (%)."""
    n = norm or norma_viab()
    return CMAP_VIAB(n(np.clip(v, n.vmin, n.vmax)))

# ---------------------------------------------------------------------------
# Matplotlib / Seaborn global style  (paper profile, Q1 journal)
# ---------------------------------------------------------------------------
def apply_style():
    """Apply the publication-quality plotting style."""
    warnings.filterwarnings("ignore")
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
    })

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def savefig(fig, name):
    """Save a figure as both PDF (vector) and PNG (raster 300 dpi)."""
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=300)
