"""
etapa4_dosis_respuesta.py – Stage 4: Dose-response modelling (4PL),
IC50 estimation with bootstrap CIs, and dose-response curve plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .config import (
    RANDOM_STATE, ORDEN_LINEAS, COLOR_LINEA,
    savefig, apply_style,
)


# ── 4PL model ─────────────────────────────────────────────────────────
def logistica_4p(x, inf, sup, ic50, hill):
    """Four-parameter logistic function."""
    return inf + (sup - inf) / (1 + (x / ic50) ** hill)


def _conc_al_50(popt):
    """Concentration at which the fitted 4PL curve crosses 50 % viability.
    Solves y(x)=50 by inverting the model; NOT popt[2] (that parameter is the
    midpoint between asymptotes, which coincides with 50 % only if inf~=0 and
    sup~=100). Returns NaN if the curve never crosses 50 % within the fitted
    asymptotes."""
    inf, sup, ic50, hill = popt
    ratio = (sup - 50.0) / (50.0 - inf)
    if ratio <= 0 or not np.isfinite(ratio):
        return np.nan
    return ic50 * ratio ** (1.0 / hill)


def _ajustar_4pl(sub):
    """Fit 4PL to a treatment x cell-line subset (raw, replicate-level rows).
    Returns dict with the point estimate, fit status, and fitted params."""
    d = sub[sub.concentracion > 0]
    x = d.concentracion.values.astype(float)
    y = d.viabilidad_pct.values.astype(float)
    n_dosis = len(np.unique(x))

    if n_dosis < 4:
        return {"IC50_ug_mL": np.nan, "estado": "insufficient (< 4 doses)",
                "n_puntos": len(d), "n_dosis": n_dosis, "popt": None}

    p0 = [min(y.min(), 20), max(y.max(), 100), np.median(x), 1.5]
    bounds = ([-10, 50, x.min() * 0.01, 0.1], [60, 150, x.max() * 100, 10])
    try:
        popt, _ = curve_fit(logistica_4p, x, y, p0=p0, bounds=bounds, maxfev=20_000)
    except Exception:
        return {"IC50_ug_mL": np.nan, "estado": "fit failure",
                "n_puntos": len(d), "n_dosis": n_dosis, "popt": None}

    ic = _conc_al_50(popt)
    base = {"n_puntos": len(d), "n_dosis": n_dosis, "popt": popt}
    if not np.isfinite(ic):
        return {**base, "IC50_ug_mL": np.nan, "estado": "does not reach 50 %"}
    if ic < x.min():
        return {**base, "IC50_ug_mL": ic, "estado": f"< {x.min():g} (lower bound)"}
    if ic > x.max():
        return {**base, "IC50_ug_mL": ic, "estado": "upper bound"}
    return {**base, "IC50_ug_mL": ic, "estado": "estimated"}


def _bootstrap_ic50(sub, nboot=300, gen=None):
    """95 % CI of the IC50 crossing by resampling replicates with replacement,
    refitting the same 4PL model each time. Uses a local RNG so it does not
    perturb the global random state."""
    gen = gen if gen is not None else np.random.default_rng(RANDOM_STATE)
    d = sub[sub.concentracion > 0]
    vals = []
    for _ in range(nboot):
        idx = gen.choice(len(d), len(d), replace=True)
        r = _ajustar_4pl(d.iloc[idx])
        if r["estado"] == "estimated":
            vals.append(r["IC50_ug_mL"])
    if len(vals) >= 20:
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return np.nan, np.nan


def estimar_ic50(dataset_limpio, activos):
    """
    Run the 4PL fit (+ bootstrap CI) for every treatment x cell-line
    combination. Returns a tidy DataFrame with one row per combination.
    """
    gen = np.random.default_rng(RANDOM_STATE)
    filas = []
    for tto in activos:
        for lc in ORDEN_LINEAS:
            sub = dataset_limpio[
                (dataset_limpio.tratamiento == tto) &
                (dataset_limpio.linea_celular == lc)
            ]
            if len(sub) == 0:
                continue
            res = _ajustar_4pl(sub)
            lo, hi = (np.nan, np.nan)
            if res["estado"] == "estimated":
                lo, hi = _bootstrap_ic50(sub, gen=gen)
            filas.append({
                "tratamiento": tto,
                "linea_celular": lc,
                "n_lecturas": res["n_puntos"],
                "IC50_ug_mL": res["IC50_ug_mL"],
                "estado": res["estado"],
                "IC50_IC95_inf": lo,
                "IC50_IC95_sup": hi,
            })

    tabla = pd.DataFrame(filas)
    print(f"IC50 table: {len(tabla)} fits")
    print(tabla[["tratamiento", "linea_celular", "IC50_ug_mL", "estado"]]
          .to_string(index=False))
    return tabla


# ── Dose-response curve plot ──────────────────────────────────────────
def plot_curvas(dataset_limpio, tratamientos, filename, titulo=None):
    """Plot dose-response curves for a list of treatments. Fits are computed
    on the raw replicate-level readings (matching estimar_ic50); points shown
    are the per-concentration mean +/- SD for readability."""
    apply_style()
    n = len(tratamientos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8), sharey=True, squeeze=False)
    axes = axes.ravel()

    for idx, tto in enumerate(tratamientos):
        ax = axes[idx]
        estados_t = []
        for lc in ORDEN_LINEAS:
            sub = dataset_limpio[
                (dataset_limpio.tratamiento == tto) &
                (dataset_limpio.linea_celular == lc) &
                (dataset_limpio.concentracion > 0)
            ]
            if len(sub) == 0:
                continue

            resumen = sub.groupby("concentracion").viabilidad_pct.agg(["mean", "std"])
            ax.errorbar(resumen.index.values, resumen["mean"].values,
                        yerr=resumen["std"].fillna(0).values, fmt="o", ms=5,
                        capsize=2.5, color=COLOR_LINEA[lc], alpha=0.9, zorder=3,
                        elinewidth=1, label=lc)

            r = _ajustar_4pl(sub)
            estados_t.append(r["estado"])
            if r["popt"] is not None:
                x_fit = np.logspace(np.log10(sub.concentracion.min()),
                                     np.log10(sub.concentracion.max()), 200)
                ax.plot(x_fit, logistica_4p(x_fit, *r["popt"]), "-",
                        color=COLOR_LINEA[lc], lw=2, alpha=0.85, zorder=2)
                if r["estado"] == "estimated":
                    ax.plot(r["IC50_ug_mL"], 50, "^", color=COLOR_LINEA[lc],
                            ms=9, zorder=4, markeredgecolor="white",
                            markeredgewidth=0.8)

        ax.set_xscale("log")
        ax.axhline(50, color="#555555", ls="--", lw=1)
        ax.set_title(tto, fontsize=11, fontweight="bold")
        ax.set_xlabel("Concentration (µg/mL)")
        if idx == 0:
            ax.set_ylabel("Viability (%)")
        ax.set_ylim(-5, 145)
        if "estimated" not in estados_t:
            ax.text(0.5, 0.9, "Non-estimable IC50", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color="#8A8A8A",
                    style="italic")
        ax.legend(fontsize=8, loc="upper right")
        ax.margins(0.04)

    if titulo:
        fig.suptitle(titulo, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, filename)
    plt.show()
    return fig
