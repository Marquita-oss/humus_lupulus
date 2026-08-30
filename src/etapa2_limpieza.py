"""
etapa2_limpieza.py – Stage 2: Data cleaning, outlier detection, and
viability normalisation.

Applies format corrections (date-encoded concentrations, mOD rescaling),
detects outliers via the Iglewicz-Hoaglin modified Z-score, normalises
absorbance to viability (%) relative to the solvent control, and produces
the dose-coverage heatmap and the viability box-plot.
"""

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    ORDEN_LINEAS, COLOR_ALARMA, CMAP_VIAB,
    norma_viab, color_viab, savefig, apply_style,
)


# ── Format cleaning ────────────────────────────────────────────────────
def limpiar_formato(consolidado):
    """
    Apply format corrections to the consolidated DataFrame.

    Returns (df_cleaned, stats_dict) where stats_dict records the count
    of each correction for auditability.
    """
    df = consolidado.copy()

    # (1) Remove header rows embedded as data (values that are column labels,
    # not readings, misplaced into the cell-line / treatment columns).
    mask_hdr = (
        df.linea_celular.astype(str).str.strip().isin(["Linea", "linea"]) |
        df.tratamiento.astype(str).str.strip().isin(["TTO", "tto", "Tratamiento"])
    )
    n_hdr = int(mask_hdr.sum())
    df = df[~mask_hdr].copy()

    # (2) Reconstruct concentrations that spreadsheet software auto-converted
    # to a date (e.g. "12,5" -> 12-May), by mapping the resulting month-day
    # back to the expected serial-dilution value; otherwise parse the decimal
    # separator normally.
    def _conc_a_numero(x):
        if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
            md = pd.Timestamp(x).strftime("%m-%d")
            return {"05-12": 12.5, "05-06": 6.25}.get(md, np.nan)
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return np.nan

    n_fecha = int(
        df.concentracion.apply(
            lambda x: isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date))
        ).sum()
    )
    df["concentracion"] = df.concentracion.apply(_conc_a_numero)

    # (3) Rescale milli-OD absorbances (values > 10 are in mOD)
    df["absorbancia"] = pd.to_numeric(df.absorbancia, errors="coerce")
    n_escala = int((df.absorbancia > 10).sum())
    df["absorbancia"] = np.where(df.absorbancia > 10, df.absorbancia / 1000, df.absorbancia)

    # (4) Standardise cell-line / treatment strings, and solvent-control dose
    df["linea_celular"] = df.linea_celular.astype(str).str.strip().replace({"MCF7": "MCF-7"})
    df["tratamiento"] = df.tratamiento.astype(str).str.strip()
    df.loc[(df.tratamiento == "CS") & (df.concentracion == 100), "concentracion"] = 0.0

    # (5) Keep only known cell lines with valid absorbance
    df = df[df.linea_celular.isin(ORDEN_LINEAS) & df.absorbancia.notna()].copy()

    # (6) Equalise dose range across cell lines (≤ 25 µg/mL): only MCF-7 was
    # tested above this window, so trimming it keeps all four lines on the
    # same dose range for PCA/clustering/modelling and for IC50 estimation.
    n_dosis_alta = int((df.concentracion > 25).sum())
    df = df[df.concentracion <= 25].copy()

    stats = {
        "headers_removed": n_hdr,
        "date_concentrations_fixed": n_fecha,
        "mod_rescaled": n_escala,
        "high_dose_removed": n_dosis_alta,
        "rows_after_cleaning": len(df),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return df, stats


# ── Outlier detection ──────────────────────────────────────────────────
def detectar_outliers(df):
    """
    Mark outliers using the Iglewicz-Hoaglin modified Z-score (|M| > 3.5).

    Returns the DataFrame with an 'es_outlier' boolean column.
    """
    def _z_modificado(s):
        x = s.values.astype(float)
        if len(x) < 4:
            return np.zeros(len(x), dtype=bool)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.zeros(len(x), dtype=bool)
        return np.abs(0.6745 * (x - med) / mad) > 3.5

    df = df.copy()
    df["_grp"] = (
        df.linea_celular + "|" + df.tratamiento + "|" +
        df.concentracion.astype(str) + "|" + df.placa.astype(str)
    )
    df["es_outlier"] = (
        df.groupby("_grp")["absorbancia"]
        .transform(_z_modificado)
        .astype(bool)
    )
    tam = df.groupby("_grp").size()
    evaluable = int((tam >= 4).sum())
    total_grp = int(tam.size)

    df = df.drop(columns="_grp").reset_index(drop=True)
    n_out = int(df.es_outlier.sum())
    print(f"Outliers flagged: {n_out} of {len(df)} ({100 * n_out / len(df):.1f} %)")
    print(f"Groups evaluable (≥ 4 replicates): {evaluable} of {total_grp}")

    return df


# ── Viability normalisation ───────────────────────────────────────────
def normalizar_viabilidad(df):
    """
    Compute viability (%) relative to solvent control (SC) per plate.

    Returns (base, cs_por_placa, cs_por_linea) where base is the
    non-outlier subset with 'viabilidad_pct' added.
    """
    base = df[~df.es_outlier].copy()
    cs_por_placa = base[base.tratamiento == "CS"].groupby(
        ["linea_celular", "placa"]
    ).absorbancia.mean()
    cs_por_linea = base[base.tratamiento == "CS"].groupby("linea_celular").absorbancia.mean()

    def _referencia_cs(row):
        clave = (row.linea_celular, row.placa)
        if clave in cs_por_placa.index and pd.notna(cs_por_placa[clave]):
            return cs_por_placa[clave], "plate SC"
        return cs_por_linea[row.linea_celular], "cell-line SC (fallback)"

    ref = base.apply(_referencia_cs, axis=1, result_type="expand")
    base["cs_ref"], base["metodo_norm"] = ref[0], ref[1]
    base["viabilidad_pct"] = base.absorbancia / base.cs_ref * 100

    print("Normalisation method used:")
    print(base.metodo_norm.value_counts().to_string())

    # QC metrics
    cs_abs = base[base.tratamiento == "CS"].groupby("placa").absorbancia.mean()
    cv_cs = cs_abs.std() / cs_abs.mean() * 100
    print(f"\nSC absorbance CV across plates: {cv_cs:.1f} % (n={cs_abs.size} plates)")

    return base, cs_por_placa, cs_por_linea


# ── Plotting: dose-coverage heatmap ───────────────────────────────────
def plot_cobertura(consolidado):
    """Stage-1 figure: dose-coverage heatmap."""
    apply_style()
    tmp = consolidado.copy()
    tmp["conc"] = pd.to_numeric(
        tmp.concentracion.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    tmp.loc[tmp.tratamiento == "CS", "conc"] = 0.0
    tmp = tmp[tmp["conc"] <= 25]
    cobertura = (
        tmp.dropna(subset=["conc"])
        .groupby(["linea_celular", "conc"])
        .size()
        .unstack(fill_value=0)
        .reindex(ORDEN_LINEAS)
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    mascara = cobertura == 0
    anot = cobertura.astype(object).where(~mascara, "—")
    sns.heatmap(
        cobertura, mask=mascara, annot=anot, fmt="", cmap="crest",
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Number of readings"},
        annot_kws={"fontsize": 10}, ax=ax,
    )
    ax.set_facecolor("#F5F5F5")
    ax.set_title("Dose coverage: readings per cell line and concentration (≤ 25 µg/mL)")
    ax.set_xlabel("Concentration (µg/mL)")
    ax.set_ylabel("")
    ax.set_xticklabels([f"{float(c):g}" for c in cobertura.columns], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    savefig(fig, "etapa1_cobertura_dosis")
    plt.show()
    return fig


# ── Plotting: viability box-plot ──────────────────────────────────────
def plot_viabilidad(df, base):
    """Stage-2 figure: viability distribution with outlier overlay."""
    apply_style()

    # Recompute viability for ALL rows (including outliers) for overlay
    cs_pp = base[base.tratamiento == "CS"].groupby(
        ["linea_celular", "placa"]
    ).absorbancia.mean()
    cs_pl = base[base.tratamiento == "CS"].groupby("linea_celular").absorbancia.mean()

    def _ref(row):
        clave = (row.linea_celular, row.placa)
        if clave in cs_pp.index and pd.notna(cs_pp[clave]):
            return cs_pp[clave]
        return cs_pl[row.linea_celular]

    todo = df.copy()
    todo["viabilidad_pct"] = todo.absorbancia / todo.apply(_ref, axis=1) * 100
    conservadas = todo[~todo.es_outlier]
    descartadas = todo[todo.es_outlier]

    orden_tto = (
        base.groupby("tratamiento").viabilidad_pct.median()
        .sort_values().index.tolist()
    )
    norm_v = norma_viab(vmin=0, vcenter=100, vmax=175)
    medianas = conservadas.groupby("tratamiento").viabilidad_pct.median()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.boxplot(
        data=conservadas, x="tratamiento", y="viabilidad_pct",
        order=orden_tto, ax=ax, width=0.45, fliersize=0, showmeans=False,
        boxprops=dict(alpha=0.85, edgecolor="#222222", linewidth=0.8),
        whiskerprops=dict(color="#444444", linewidth=0.8),
        capprops=dict(color="#444444", linewidth=0.8),
        medianprops=dict(color="#222222", linewidth=1.0),
    )
    for parche, t in zip(ax.patches, orden_tto):
        parche.set_facecolor(color_viab(medianas[t], norm_v))

    sns.stripplot(
        data=conservadas, x="tratamiento", y="viabilidad_pct",
        order=orden_tto, ax=ax, color="#333333", size=2.0,
        alpha=0.25, jitter=0.25,
    )

    pos = {t: i for i, t in enumerate(orden_tto)}
    rng = np.random.default_rng(42)
    xo = descartadas.tratamiento.map(pos) + rng.uniform(-0.22, 0.22, len(descartadas))
    ax.scatter(
        xo, descartadas.viabilidad_pct, marker="x", s=34, linewidths=1.3,
        color=COLOR_ALARMA, zorder=5, label=f"discarded outlier (n={len(descartadas)})",
    )
    ax.axhline(100, color="grey", ls="-", lw=1)
    ax.axhline(50, color="#555555", ls="--", lw=1.2)
    ax.text(len(orden_tto) - 0.5, 102, "SC control = 100 %", color="grey",
            fontsize=8, va="bottom", ha="right")
    ax.text(len(orden_tto) - 0.5, 52, "cytotoxic threshold 50 %", color="#555555",
            fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Viability (%)")
    ax.set_title("Retained viability by treatment and discarded outliers")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=CMAP_VIAB, norm=norm_v)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Median viability (%)")
    cb.set_ticks([0, 50, 100, 150])
    ax.set_xticklabels(
        ["SC" if t == "CS" else "NC" if t == "C-" else t for t in orden_tto]
    )
    savefig(fig, "etapa2_limpieza")
    plt.show()
    return fig


# ── Export clean dataset ──────────────────────────────────────────────
def exportar_dataset_limpio(base, output_dir):
    """Save the clean dataset to CSV."""
    import os
    cols = [
        "linea_celular", "placa", "tratamiento", "concentracion",
        "absorbancia", "viabilidad_pct", "cs_ref", "metodo_norm", "es_outlier",
    ]
    dataset = base.reset_index(drop=True)[[c for c in cols if c in base.columns]]
    path = os.path.join(output_dir, "dataset_limpio_final.csv")
    dataset.to_csv(path, index=False)
    print(f"Saved: {path}  {dataset.shape}")
    return dataset
