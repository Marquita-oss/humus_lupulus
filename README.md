# A Machine Learning Approach for the Characterization of the Selective Antitumor Activity of Cold-Extracted *Humulus lupulus* in Breast and Gastric Cancer Cell Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the reproducible data-science pipeline for evaluating the cytotoxic and selective antitumor activity of cold-extracted *Humulus lupulus* (hop) extracts on human cancer cell lines (MCF-7, MDA-MB-231, AGS) and the non-tumour reference line GES-1.

The analysis covers six stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `src/etapa1_consolidacion.py` | Raw Excel ingestion and tidy consolidation |
| 2 | `src/etapa2_limpieza.py` | Format cleaning, outlier detection (Iglewicz-Hoaglin), viability normalisation |
| 3 | `src/etapa3_pca.py` | PCA on the treatment × cell-line viability matrix |
| 4 | `src/etapa4_dosis_respuesta.py` | 4-parameter log-logistic dose-response model, IC₅₀ with bootstrap CI |
| 5 | `src/etapa5_modelos.py` | Clustering (k-means, Ward), classification (DT, RF, SVM, LR, MLP), SHAP |
| 6 | `src/etapa6_selectividad.py` | Selectivity Index computation and ranking |

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-user>/humus_lupulus.git
cd humus_lupulus

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the full pipeline
python scripts/run_pipeline.py --data /path/to/Datos_Todo.xlsx
```

Running the pipeline saves figures (PDF + PNG at 300 dpi) to `figures/` and processed data tables to `output/` on your machine. These two folders are git-ignored and are **not** published in this repository: the underlying viability and IC50 values are sensitive research data, so only the analysis code is shared here. Anyone with access to the raw data can regenerate every figure and table locally by running the pipeline as above.

## Data Availability

The raw experimental data (SRB absorbance readings) were generated at the Centro de Investigaciones Biomédicas, Universidad de Valparaíso. The dataset, and the processed tables and figures derived from it, are available upon reasonable request to the corresponding author.

## Project Structure

```
humus_lupulus/
├── src/                        # Modular analysis pipeline
│   ├── __init__.py
│   ├── config.py               # Constants, palettes, style
│   ├── etapa1_consolidacion.py # Stage 1: Data ingestion
│   ├── etapa2_limpieza.py      # Stage 2: Cleaning & normalisation
│   ├── etapa3_pca.py           # Stage 3: PCA
│   ├── etapa4_dosis_respuesta.py # Stage 4: Dose-response (4PL)
│   ├── etapa5_modelos.py       # Stage 5: ML models & SHAP
│   └── etapa6_selectividad.py  # Stage 6: Selectivity Index
├── scripts/
│   └── run_pipeline.py         # Master pipeline script
├── figures/                    # Generated figures (PDF + PNG) -- git-ignored, not published
├── output/                     # Processed CSV tables -- git-ignored, not published
├── notebooks/                  # Exploratory notebooks (optional)
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

## Colour Palette

All figures follow a unified colour system designed for colour-blind accessibility and Q1 journal standards:

- **Cell lines**: Paul Tol Vibrant palette (MCF-7 blue, MDA teal, AGS orange, GES-1 magenta)
- **Viability heatmaps**: RdBu divergent, centred at 100 % (solvent control)
- **Categorical heatmaps**: `crest` (seaborn)
- **Binary groups**: Navy `#28417a` + Crimson `#C1272D`
- **SHAP grouped importance**: `crest` gradient

## Citation

If you use this code in your research, please cite:

> Marca R., Báez C., Villena J., Roldán N., Salas R. *A Machine Learning Approach for the Characterization of the Selective Antitumor Activity of Cold-Extracted Humulus lupulus in Breast and Gastric Cancer Cell Models.* (2026).

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
