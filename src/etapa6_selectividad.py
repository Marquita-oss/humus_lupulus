"""
etapa6_selectividad.py – Stage 6: Selectivity Index calculation and plot.

SI = IC50(GES-1) / IC50(tumour line).  Only pairs with both IC50 values
estimated are included.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import (
    COLOR_LINEA, COLOR_ALARMA, LINEAS_TUMORALES,
    savefig, apply_style, OUTPUT_DIR,
)


def calcular_indice_selectividad(tabla_ic50):
    """
    Compute the Selectivity Index for all valid treatment × tumour-line pairs.
    Returns (tabla_is, resumen_is).
    """
    estimados = tabla_ic50[tabla_ic50.estado == "estimated"]
    piv = estimados.pivot_table(
        index="tratamiento", columns="linea_celular", values="IC50_ug_mL"
    )

    filas = []
    for t in piv.index:
        if "GES-1" not in piv.columns or pd.isna(piv.loc[t, "GES-1"]):
            continue
        ges = piv.loc[t, "GES-1"]
        for lc in LINEAS_TUMORALES:
            if lc in piv.columns and pd.notna(piv.loc[t, lc]):
                filas.append({
                    "tratamiento": t,
                    "linea_tumoral": lc,
                    "IC50_sana_GES1": ges,
                    "IC50_tumoral": piv.loc[t, lc],
                    "indice_selectividad": ges / piv.loc[t, lc],
                })

    tabla_is = pd.DataFrame(filas).sort_values("indice_selectividad", ascending=False)
    tabla_is.to_csv(f"{OUTPUT_DIR}/tabla_indice_selectividad.csv", index=False)

    resumen = (
        tabla_is.groupby("tratamiento")
        .indice_selectividad.mean()
        .sort_values(ascending=False)
        .round(2)
    )
    print("Treatments with computable SI:", tabla_is.tratamiento.unique().tolist())
    print("\nSelectivity Index per treatment × tumour line:")
    print(tabla_is.round(2).to_string(index=False))
    print("\nRanking by mean SI:", " > ".join(f"{t} ({v})" for t, v in resumen.items()))

    return tabla_is, resumen


def plot_selectividad(tabla_is, resumen_is):
    """Stage-6 figure: horizontal bar chart of Selectivity Index."""
    apply_style()

    orden_tto = resumen_is.index.tolist()[::-1]
    filas = []
    for t in orden_tto:
        sub = tabla_is[tabla_is.tratamiento == t].sort_values("indice_selectividad")
        for _, row in sub.iterrows():
            filas.append(row)
    plot_df = pd.DataFrame(filas).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(plot_df) + 1.8))
    ypos = range(len(plot_df))
    colores = [COLOR_LINEA[lc] for lc in plot_df.linea_tumoral]
    ax.barh(ypos, plot_df.indice_selectividad, color=colores, height=0.7,
            zorder=2, edgecolor="white", lw=0.6)

    for i, v in enumerate(plot_df.indice_selectividad):
        ax.annotate(f"{v:.1f}", (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=10, fontweight="bold")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(plot_df.linea_tumoral)

    # Treatment group labels
    from itertools import groupby
    i0 = 0
    for t, grp in groupby(plot_df.tratamiento):
        n = len(list(grp))
        centro = i0 + (n - 1) / 2
        ax.text(-0.08, centro, t, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12, fontweight="bold")
        if i0 > 0:
            ax.axhline(i0 - 0.5, color="#D9D9D9", lw=0.8, zorder=0)
        i0 += n

    ax.axvline(1, color=COLOR_ALARMA, ls="--", lw=1.5, zorder=3)
    ax.set_xlim(0, plot_df.indice_selectividad.max() * 1.15)
    ax.set_ylim(-0.6, len(plot_df) - 0.4)
    ax.set_title("Selectivity Index by Treatment and Tumor Cell Line", pad=22)
    ax.annotate("SI = 1 (no selectivity)", (1, len(plot_df) - 0.4),
                xytext=(4, 8), textcoords="offset points",
                color=COLOR_ALARMA, fontsize=8, va="bottom")
    ax.set_xlabel(
        "Selectivity Index = healthy IC₅₀ (GES-1) ÷ tumor IC₅₀   (>1 = selective towards tumor)"
    )
    ax.set_ylabel("")

    tumoral_in = [lc for lc in LINEAS_TUMORALES if lc in plot_df.linea_tumoral.values]
    ax.legend(
        handles=[mpatches.Patch(color=COLOR_LINEA[lc], label=lc) for lc in tumoral_in],
        title="Tumor cell line", loc="lower right", frameon=False,
    )
    fig.tight_layout()
    savefig(fig, "etapa6_selectividad")
    plt.show()
    return fig
