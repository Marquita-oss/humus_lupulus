#!/usr/bin/env python3
"""
run_pipeline.py – Master script that executes the full six-stage
cytotoxicity analysis pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py --data path/to/Datos_Todo.xlsx

All outputs (figures as PDF+PNG, CSV tables) are written to the
`figures/` and `output/` directories.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add the parent directory to sys.path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RANDOM_STATE, ACTIVOS, OUTPUT_DIR, apply_style
from src.etapa1_consolidacion import consolidar, verificar_fidelidad
from src.etapa2_limpieza import (
    limpiar_formato, detectar_outliers, normalizar_viabilidad,
    plot_cobertura, plot_viabilidad, exportar_dataset_limpio,
)
from src.etapa3_pca import construir_matriz, ejecutar_pca, plot_pca
from src.etapa4_dosis_respuesta import estimar_ic50, plot_curvas
from src.etapa5_modelos import (
    clustering, lda_histograma, preparar_clasificacion,
    entrenar_clasificadores, plot_roc_confusion, plot_mlp_loss, plot_shap,
)
from src.etapa6_selectividad import calcular_indice_selectividad, plot_selectividad


def main(ruta_xlsx: str):
    """Run all six pipeline stages."""
    # Ensure reproducibility
    np.random.seed(RANDOM_STATE)
    apply_style()

    print("=" * 60)
    print("STAGE 1 – Data consolidation")
    print("=" * 60)
    consolidado, linea_de = consolidar(ruta_xlsx)

    print("\n" + "=" * 60)
    print("STAGE 2 – Cleaning, outliers, viability normalisation")
    print("=" * 60)
    df_limpio, stats = limpiar_formato(consolidado)
    df_limpio = detectar_outliers(df_limpio)
    base, cs_pp, cs_pl = normalizar_viabilidad(df_limpio)

    # Figures
    plot_cobertura(consolidado)
    plot_viabilidad(df_limpio, base)

    # Export
    dataset_limpio = exportar_dataset_limpio(base, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("STAGE 3 – PCA")
    print("=" * 60)
    matriz = construir_matriz(dataset_limpio)
    scores, var, X_std = ejecutar_pca(matriz)
    plot_pca(matriz, scores, var)

    print("\n" + "=" * 60)
    print("STAGE 4 – Dose-response modelling (4PL) & IC50")
    print("=" * 60)
    tabla_ic50 = estimar_ic50(dataset_limpio, ACTIVOS)
    tabla_ic50.to_csv(os.path.join(OUTPUT_DIR, "tabla_ic50.csv"), index=False)

    plot_curvas(dataset_limpio, ["AT", "T3", "PN"], "etapa4_dosis_respuesta",
                titulo="Dose-response: main extracts")
    plot_curvas(dataset_limpio, ["F2", "F3", "F4"], "etapa4_fracciones",
                titulo="Dose-response: T3 fractions")

    print("\n" + "=" * 60)
    print("STAGE 5 – Clustering, classification & SHAP")
    print("=" * 60)
    # 5a. Clustering
    from sklearn.cluster import KMeans
    k_opt, sil = clustering(X_std, matriz)
    labels = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE, n_init=10).fit_predict(X_std)
    lda_histograma(X_std, labels, matriz)

    # 5b. Classification
    obs = preparar_clasificacion(dataset_limpio)
    resultados, pre, X, y, mejor = entrenar_clasificadores(obs)
    plot_roc_confusion(resultados, y, mejor)
    plot_mlp_loss(pre, X, y)
    plot_shap(pre, X, y)

    tabla_clasificadores = pd.DataFrame([
        {"modelo": name, "exactitud": res["acc"], "f1": res["f1"], "auc": res["auc"]}
        for name, res in resultados.items()
    ]).sort_values("f1", ascending=False)
    tabla_clasificadores.to_csv(
        os.path.join(OUTPUT_DIR, "tabla_clasificadores.csv"), index=False
    )

    print("\n" + "=" * 60)
    print("STAGE 6 – Selectivity Index")
    print("=" * 60)
    tabla_is, resumen_is = calcular_indice_selectividad(tabla_ic50)
    plot_selectividad(tabla_is, resumen_is)

    # Final summary
    print("\n" + "=" * 60)
    print("QUANTITATIVE SUMMARY")
    print("=" * 60)
    n_citotox = (
        dataset_limpio[dataset_limpio.tratamiento.isin(["AT", "T3", "PN"])]
        .groupby("tratamiento").viabilidad_pct.mean().round(1)
    )
    ranking = " > ".join(f"{t} (SI mean {v})" for t, v in resumen_is.items())
    print("Mean viability (%) of main extracts:")
    print(n_citotox.to_string())
    print(f"\nBest classifier: {mejor} "
          f"(acc={resultados[mejor]['acc']:.3f}, F1={resultados[mejor]['f1']:.3f})")
    print(f"Clusters: k={k_opt}, silhouette={sil[k_opt]:.3f}")
    print(f"SI ranking: {ranking}")
    print("\n✅  Pipeline completed. Figures in figures/ · Tables in output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Humulus lupulus cytotoxicity analysis pipeline."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the raw Excel workbook (e.g. 'Copia de Datos_Todo.xlsx')"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    main(args.data)
