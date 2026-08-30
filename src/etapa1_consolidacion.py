"""
etapa1_consolidacion.py – Stage 1: Data ingestion and consolidation.

Reads the raw Excel workbook (heterogeneous sheets), extracts absorbance
readings from wide-format and long-format sources, and produces a tidy
DataFrame with one row per reading and neutral plate IDs.
"""

import numpy as np
import pandas as pd


# ── Sheet-to-cell-line mapping ──────────────────────────────────────────
_LINEAS_II = {
    "II Parte - MDA":   "MDA",
    "II Parte - GES-1": "GES-1",
    "II Parte - AGS":   "AGS",
    "II Parte - MCF-7": "MCF-7",
}

FUENTES_DATOS = [
    "Tabla madre",
    "II Parte - MCF-7",
    "II Parte - GES-1",
    "II Parte - AGS",
    "II Parte - MDA",
]


def _parsear_formato_ancho(ruta_xlsx, hoja, linea):
    """Extract readings from a wide-format sheet (plate blocks A–H × 12 doses)."""
    d = pd.read_excel(ruta_xlsx, sheet_name=hoja, engine="openpyxl", header=None)
    filas_dosis = []
    for i in range(len(d)):
        vals = d.iloc[i, 1:13].tolist()
        try:
            nums = [float(x) for x in vals]
            if not any(np.isnan(nums)) and 25 in [round(x, 3) for x in nums]:
                filas_dosis.append((i, nums))
        except Exception:
            pass

    filas = []
    for bloque, (r, dosis) in enumerate(filas_dosis):
        inicio = next(
            (k for k in range(r + 1, min(r + 4, len(d)))
             if str(d.iloc[k, 0]).strip() == "A"),
            None,
        )
        if inicio is None:
            continue
        j = inicio
        while j < len(d) and str(d.iloc[j, 0]).strip() in list("ABCDEFGH"):
            tto = str(d.iloc[j, 13]).strip()
            if tto.startswith("AT-"):
                tto = "AT"
            if tto not in ("", "nan", "NaN"):
                for col in range(1, 13):
                    val = pd.to_numeric(d.iloc[j, col], errors="coerce")
                    if pd.notna(val):
                        filas.append({
                            "linea_celular": linea,
                            "_hoja": hoja.strip(),
                            "_bloque": f"{hoja.strip()}#{bloque}",
                            "tratamiento": tto,
                            "concentracion": float(dosis[col - 1]),
                            "absorbancia": float(val),
                        })
            j += 1
    return pd.DataFrame(filas)


def inventario_hojas(ruta_xlsx):
    """Return a DataFrame describing every sheet in the workbook."""
    xls = pd.ExcelFile(ruta_xlsx, engine="openpyxl")
    rows = []
    for hoja in xls.sheet_names:
        d = pd.read_excel(ruta_xlsx, sheet_name=hoja, header=None, engine="openpyxl")
        rows.append({
            "hoja": hoja.strip(),
            "filas": d.shape[0],
            "columnas": d.shape[1],
            "celdas_con_dato": int(d.notna().sum().sum()),
        })
    inv = pd.DataFrame(rows)

    def rol(h):
        if h in FUENTES_DATOS:
            return "DATA SOURCE"
        if h.startswith("IC50"):
            return "IC50 reference (team)"
        if "SELECTIVIDAD" in h.upper():
            return "selectivity formula"
        if h in ("Reuniones", "INFOS"):
            return "team notes"
        return "documentation / design"

    inv["rol"] = inv["hoja"].apply(rol)
    return inv


def consolidar(ruta_xlsx):
    """
    Main entry point.  Returns the consolidated tidy DataFrame with columns:
    [linea_celular, placa, tratamiento, concentracion, absorbancia].
    """
    hojas_excel = pd.ExcelFile(ruta_xlsx, engine="openpyxl").sheet_names
    linea_de = {h: _LINEAS_II[h.strip()] for h in hojas_excel if h.strip() in _LINEAS_II}

    # Wide-format sheets
    anchas = pd.concat(
        [_parsear_formato_ancho(ruta_xlsx, h, l) for h, l in linea_de.items()],
        ignore_index=True,
    )

    # Long-format sheet ('Tabla madre')
    larga = pd.read_excel(ruta_xlsx, sheet_name="Tabla madre", engine="openpyxl")
    larga.columns = ["linea_celular", "_placa_orig", "tratamiento", "concentracion", "absorbancia"]
    larga["_hoja"] = "Tabla madre"
    larga["_bloque"] = "TM#" + larga["_placa_orig"].astype(str)
    larga = larga.drop(columns="_placa_orig")

    consolidado = pd.concat([larga, anchas], ignore_index=True)

    # Fix known concentration mislabelling: the raw dose-header row records the
    # integer 5 in the "II Parte - MDA" and "II Parte - GES-1" sheets, in the exact
    # structural position (4 replicate columns, between the 25 and 1 ug/mL blocks)
    # where "II Parte - MCF-7" and "II Parte - AGS" record 0.5. No 0.5 value appears
    # anywhere else in the MDA/GES-1 sheets, and the intended protocol dose series is
    # 25/12.5/6.25/1/0.5 across all four lines. This is consistent with a decimal-point
    # transcription slip (0.5 -> 5) specific to those two sheets, not with any of the
    # other documented format artifacts (date auto-conversion, decimal separator, OD
    # rescaling). Treated here as a correction on that basis; confirm against the bench
    # notebook before relying on it for a final submission.
    consolidado["concentracion"] = consolidado["concentracion"].replace(
        {5.0: 0.5, 5: 0.5, "5.0": 0.5, "5": 0.5}
    )

    # Neutral correlative plate IDs (not encoding sheet of origin)
    bloques = {
        b: f"P{n + 1:02d}"
        for n, b in enumerate(sorted(consolidado["_bloque"].dropna().unique()))
    }
    consolidado["placa"] = consolidado["_bloque"].map(bloques)
    consolidado = consolidado.drop(columns=["_hoja", "_bloque"])
    consolidado = consolidado[["linea_celular", "placa", "tratamiento", "concentracion", "absorbancia"]]

    print(f"Consolidated table: {consolidado.shape[0]} rows × {consolidado.shape[1]} columns")
    print(f"Neutral plate IDs: {consolidado.placa.nunique()} "
          f"(P01 … P{consolidado.placa.nunique():02d})")

    return consolidado, linea_de


def verificar_fidelidad(ruta_xlsx, anchas_df, linea_de):
    """Check that every parsed absorbance value exists in the raw sheet grid."""
    fidelidad = []
    for hoja, linea in linea_de.items():
        crudo = pd.read_excel(ruta_xlsx, sheet_name=hoja, engine="openpyxl", header=None)
        valores_crudos = set(np.round(pd.to_numeric(crudo.values.ravel(), errors="coerce"), 4))
        valores_crudos.discard(np.nan)
        ext = anchas_df[anchas_df.linea_celular == linea].absorbancia
        presentes = ext.round(4).isin(valores_crudos).mean() * 100
        fidelidad.append({
            "hoja": hoja.strip(),
            "linea": linea,
            "n_extraidas": len(ext),
            "pct_halladas": round(presentes, 1),
        })
    fid = pd.DataFrame(fidelidad)
    print(fid.to_string(index=False))
    print(f"\nGlobal extraction fidelity: {fid['pct_halladas'].min():.1f} % (minimum across sheets)")
    return fid
