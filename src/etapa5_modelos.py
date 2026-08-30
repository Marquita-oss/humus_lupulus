"""
etapa5_modelos.py – Stage 5: Clustering, classification, LDA,
ROC curves, MLP loss curve, and SHAP importance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from itertools import groupby as itertools_groupby
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import (
    RANDOM_STATE, ORDEN_LINEAS, ACTIVOS, CMAP_VIAB,
    COLOR_LINEA, COLOR_ALARMA, COLOR_CITOTOXICO, COLOR_BAJA_CITOT,
    PALETA_MODELOS, ESTILO_MODELOS,
    savefig, apply_style,
)


# ══════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════
def clustering(X_std, matriz):
    """K-means silhouette sweep + Ward dendrogram.  Returns (k_opt, sil_dict)."""
    apply_style()
    sil = {
        k: silhouette_score(
            X_std,
            KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_std),
        )
        for k in range(2, min(len(X_std), 8))
    }
    k_opt = max(sil, key=sil.get)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) Silhouette
    axes[0].bar(list(sil.keys()), list(sil.values()), color="#28417a", width=0.55)
    axes[0].set_title("a · Silhouette coefficient by k")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_xticks(list(sil.keys()))

    # (b) Dendrogram
    Z = linkage(X_std, method="ward")
    set_link_color_palette([COLOR_BAJA_CITOT, COLOR_CITOTOXICO])
    dendrogram(
        Z, labels=matriz.index.tolist(), ax=axes[1],
        leaf_rotation=45, leaf_font_size=10,
        color_threshold=Z[-(k_opt - 1), 2],
        above_threshold_color="#888888",
    )
    set_link_color_palette(None)
    axes[1].set_title(f"b · Hierarchical clustering (Ward, k={k_opt})")
    axes[1].set_ylabel("Distance")

    savefig(fig, "etapa5_clustering")
    plt.show()
    return k_opt, sil


# ══════════════════════════════════════════════════════════════════════
# LDA
# ══════════════════════════════════════════════════════════════════════
def lda_histograma(X_std, labels, matriz):
    """Fisher's LDA projected histogram for k=2 clustering."""
    apply_style()
    lda = LinearDiscriminantAnalysis()
    z = lda.fit_transform(X_std, labels).ravel()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cls, color, label in [(0, COLOR_BAJA_CITOT, "Low cytotoxicity"),
                               (1, COLOR_CITOTOXICO, "Cytotoxic")]:
        mask = labels == cls
        ax.hist(z[mask], bins=8, alpha=0.7, color=color, label=label,
                edgecolor="white", lw=0.8)
    ax.set_title("LDA projection (k = 2 groups)")
    ax.set_xlabel("LD1 score")
    ax.set_ylabel("Count")
    ax.legend()
    savefig(fig, "etapa5_lda")
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
def preparar_clasificacion(dataset_limpio):
    """Build the classification dataset (n=152 after averaging replicates)."""
    obs = (
        dataset_limpio[
            dataset_limpio.tratamiento.isin(ACTIVOS) &
            (dataset_limpio.concentracion > 0)
        ]
        .groupby(["tratamiento", "linea_celular", "concentracion"])
        .agg(viab_media=("viabilidad_pct", "mean"))
        .reset_index()
    )
    obs["citotoxico"] = (obs.viab_media < 50).astype(int)
    print(f"Classification dataset: {len(obs)} obs, "
          f"{obs.citotoxico.sum()} cytotoxic ({100 * obs.citotoxico.mean():.1f} %)")
    return obs


def entrenar_clasificadores(obs):
    """
    Train 4 classifiers with LOO-CV.  Returns (resultados, pre, X, y).
    """
    X = obs[["tratamiento", "linea_celular", "concentracion"]]
    y = obs["citotoxico"].values

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
         ["tratamiento", "linea_celular"]),
        ("num", "passthrough", ["concentracion"]),
    ])

    modelos = {
        "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu", max_iter=2500,
            random_state=RANDOM_STATE, alpha=0.01),
    }

    pipe = {
        name: Pipeline([("pre", pre), ("clf", clf)])
        for name, clf in modelos.items()
    }

    loo = LeaveOneOut()
    resultados = {}
    for name, p in pipe.items():
        y_pred = cross_val_predict(p, X, y, cv=loo)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # ROC-AUC (need probabilities)
        try:
            y_prob = cross_val_predict(p, X, y, cv=loo, method="predict_proba")[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc = None, None, None

        resultados[name] = {
            "acc": acc, "f1": f1, "auc": roc_auc,
            "y_pred": y_pred, "fpr": fpr, "tpr": tpr,
        }
        print(f"  {name:25s}  acc={acc:.3f}  F1={f1:.3f}  AUC={roc_auc or '---'}")

    mejor = max(resultados, key=lambda n: resultados[n]["f1"])
    print(f"\nBest model: {mejor}")

    return resultados, pre, X, y, mejor


# ── ROC + Confusion Matrix plot ───────────────────────────────────────
def plot_roc_confusion(resultados, y, mejor):
    """Stage-5 figure: ROC curves + confusion matrix of best model."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) ROC
    for name, res in resultados.items():
        if res["fpr"] is not None:
            axes[0].plot(
                res["fpr"], res["tpr"],
                color=PALETA_MODELOS.get(name, "#666666"),
                linestyle=ESTILO_MODELOS.get(name, "-"),
                lw=2,
                label=f"{name} (AUC={res['auc']:.2f})",
            )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    axes[0].set_title("a · ROC curves (leave-one-out)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8, loc="lower right")

    # (b) Confusion matrix
    cm = confusion_matrix(y, resultados[mejor]["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="crest", ax=axes[1],
        xticklabels=["Not cytotoxic", "Cytotoxic"],
        yticklabels=["Not cytotoxic", "Cytotoxic"],
        linewidths=0.8, linecolor="white",
    )
    axes[1].set_title(f"b · Confusion matrix ({mejor})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    fig.tight_layout()
    savefig(fig, "etapa5_clasificadores")
    plt.show()
    return fig


# ── MLP loss curve ────────────────────────────────────────────────────
def plot_mlp_loss(pre, X, y):
    """Train MLP and plot loss curve."""
    apply_style()
    Xt = pre.fit_transform(X)
    Xt_denso = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", max_iter=2500,
        random_state=RANDOM_STATE, alpha=0.01,
    )
    mlp.fit(Xt_denso, y)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(mlp.loss_curve_, color="#28417a", lw=2)
    ax.set_title("MLP · Training loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log-loss)")
    ax.margins(0.02)
    ax.grid(True, which="both", color="#EAEAEA", linestyle=":", linewidth=0.5)
    savefig(fig, "etapa5_mlp_perdida")
    plt.show()
    return fig, Xt_denso


# ══════════════════════════════════════════════════════════════════════
# SHAP IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
def plot_shap(pre, X, y):
    """SHAP beeswarm + grouped bar importance."""
    apply_style()

    Xt = pre.fit_transform(X)
    Xt_denso = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ).fit(Xt_denso, y)

    nombres_var = pre.get_feature_names_out()
    shap_vals = np.asarray(shap.TreeExplainer(rf).shap_values(Xt_denso))
    sv = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals

    # Group importance
    def _grupo(n):
        if n.startswith("cat__tratamiento"):
            return "Treatment"
        if n.startswith("cat__linea_celular"):
            return "Cell line"
        return "Concentration"

    imp = (
        pd.Series(np.abs(sv).mean(axis=0), index=nombres_var)
        .groupby(_grupo).sum().sort_values()
    )
    imp_rel = imp / imp.sum()

    # Feature labels for beeswarm
    def _etiqueta(n):
        return (n.replace("cat__tratamiento_", "tto: ")
                 .replace("cat__linea_celular_", "line: ")
                 .replace("num__", "")
                 .replace("concentracion", "Concentration"))

    val = pd.DataFrame(Xt_denso, columns=nombres_var)
    val_norm = (val - val.min()) / (val.max() - val.min()).replace(0, 1)
    orden_ind = pd.Series(np.abs(sv).mean(axis=0), index=nombres_var).sort_values(ascending=False)
    top = orden_ind.head(8).index.tolist()[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) Beeswarm
    rng = np.random.default_rng(RANDOM_STATE)
    cmap_sw = CMAP_VIAB.reversed()
    for i, var in enumerate(top):
        j = list(nombres_var).index(var)
        x = sv[:, j]
        yy = i + rng.uniform(-0.18, 0.18, len(x))
        axes[0].scatter(x, yy, c=val_norm[var].values, cmap=cmap_sw, s=22,
                        alpha=0.75, edgecolor="none", vmin=0, vmax=1, zorder=3)
    axes[0].axvline(0, color="#888888", lw=1, zorder=1)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels([_etiqueta(v) for v in top])
    axes[0].set_title("a · Impact per observation (SHAP · Random Forest)")
    axes[0].set_xlabel("SHAP value (→ pushes to 'cytotoxic')")
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_sw, norm=plt.Normalize(0, 1)),
        ax=axes[0], fraction=0.035, pad=0.02,
    )
    cb.set_label("Feature value")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])

    # (b) Grouped importance bars
    _crest_bars = sns.color_palette("crest", n_colors=len(imp_rel))
    b = axes[1].barh(
        range(len(imp_rel)), imp_rel.values,
        color=_crest_bars, height=0.62, zorder=2,
    )
    for i, v in enumerate(imp_rel.values):
        axes[1].annotate(
            f"{v * 100:.0f} %", (v, i), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=10.5,
            fontweight="bold",
        )
    axes[1].set_yticks(range(len(imp_rel)))
    axes[1].set_yticklabels(imp_rel.index)
    axes[1].set_title("b · Grouped importance")
    axes[1].set_xlabel("Relative contribution")
    axes[1].set_xlim(0, max(imp_rel.values) * 1.28)

    fig.tight_layout()
    savefig(fig, "etapa5_shap")
    plt.show()

    print("Relative importance:",
          {k: f"{v * 100:.0f}%" for k, v in imp_rel.sort_values(ascending=False).items()})
    return fig, imp_rel
