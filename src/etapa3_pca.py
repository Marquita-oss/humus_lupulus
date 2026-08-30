"""
etapa3_pca.py – Stage 3: Principal Component Analysis on the
treatment × cell-line viability matrix.

Produces the two-panel figure: (a) scree plot and (b) PC1-PC2 scatter
coloured by mean tumour viability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import (
    ORDEN_LINEAS, LINEAS_TUMORALES, ACTIVOS, CMAP_VIAB,
    norma_viab, color_viab, savefig, apply_style,
)


def construir_matriz(dataset_limpio):
    """Build the treatment × cell-line mean-viability matrix."""
    matriz = (
        dataset_limpio[dataset_limpio.tratamiento.isin(ACTIVOS)]
        .pivot_table(
            index="tratamiento",
            columns="linea_celular",
            values="viabilidad_pct",
            aggfunc="mean",
        )[ORDEN_LINEAS]
        .dropna()
    )
    print("Treatment × cell-line matrix (mean viability %):")
    print(matriz.round(1).to_string())
    return matriz


def ejecutar_pca(matriz):
    """Standardise and run PCA.  Returns (scores, var_explained, X_std)."""
    X_std = StandardScaler().fit_transform(matriz.values)
    pca = PCA()
    scores = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_
    return scores, var, X_std


def plot_pca(matriz, scores, var):
    """Stage-3 figure: scree + PC1-PC2 scatter."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) Scree plot
    axes[0].bar(range(1, len(var) + 1), var * 100, color="#28417a", width=0.65)
    axes[0].plot(range(1, len(var) + 1), np.cumsum(var) * 100,
                 "o-", color="#404040", lw=1.5, ms=6)
    for i, v in enumerate(var, 1):
        axes[0].text(i, v * 100 + 2, f"{v * 100:.0f}%",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("a · Explained variance by component")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_xticks(range(1, len(var) + 1))
    axes[0].set_ylim(0, 105)

    # (b) Scatter PC1 vs PC2
    viab_media = matriz[LINEAS_TUMORALES].mean(axis=1)
    norm_v = norma_viab(vmin=40, vcenter=100, vmax=150)
    cols = [color_viab(v, norm_v) for v in viab_media]
    axes[1].scatter(scores[:, 0], scores[:, 1], s=160, c=cols,
                    edgecolor="#222222", lw=0.8, zorder=3)
    axes[1].grid(True, which="both", color="#EAEAEA", linestyle=":", linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=CMAP_VIAB, norm=norm_v)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("Mean tumor viability (%)")
    cb.set_ticks([40, 70, 100, 130, 150])

    for i, t in enumerate(matriz.index):
        axes[1].annotate(
            t, (scores[i, 0], scores[i, 1]),
            fontsize=10, fontweight="bold",
            xytext=(7, 4), textcoords="offset points",
        )
    axes[1].axhline(0, color="grey", lw=0.7, zorder=0)
    axes[1].axvline(0, color="grey", lw=0.7, zorder=0)
    axes[1].set_title("b · Treatments in the PC1-PC2 plane")
    axes[1].set_xlabel(f"PC1 · global cytotoxicity ({var[0] * 100:.0f} %)")
    axes[1].set_ylabel(f"PC2 ({var[1] * 100:.0f} %)")
    axes[1].margins(0.12)

    savefig(fig, "etapa3_pca")
    plt.show()
    return fig
