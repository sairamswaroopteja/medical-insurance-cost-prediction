"""
evaluate.py
-----------
Enhanced evaluation + plotting utilities.

Functions
---------
plot_pred_vs_actual   – scatter of predicted vs actual for one model
plot_residuals        – residuals vs fitted + histogram
plot_feature_importance – horizontal bar chart
plot_model_comparison   – grouped bar chart comparing all models
plot_cv_scores          – CV RMSE mean ± std for all models
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

PALETTE = {
    "LinearRegression": "#4C72B0",
    "Ridge":            "#DD8452",
    "Lasso":            "#55A868",
    "KNN":              "#F7C948",
    "RandomForest":     "#C44E52",
    "AdaBoost":         "#8172B2",
}
DEFAULT_COLOR = "#4C72B0"


# ── Per-model plots ───────────────────────────────────────────────────────────

def plot_pred_vs_actual(y_true, y_pred, title: str, save_path=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, edgecolors="k",
               linewidths=0.3, s=30,
               color=PALETTE.get(title.split("—")[0].strip(), DEFAULT_COLOR))
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Actual Charges ($)"); ax.set_ylabel("Predicted Charges ($)")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, title: str, save_path=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    residuals = y_true - y_pred
    color = PALETTE.get(title.split("—")[0].strip(), "#DD8452")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=25, color=color)
    axes[0].axhline(0, color="red", lw=1.5, ls="--")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    axes[0].set_xlabel("Predicted ($)"); axes[0].set_ylabel("Residual ($)")
    axes[0].set_title(f"{title} — Residuals vs Fitted")

    # Residual histogram
    axes[1].hist(residuals, bins=40, edgecolor="k", color=color, alpha=0.8)
    axes[1].set_xlabel("Residual ($)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title} — Residual Distribution")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, title: str, save_path=None, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_imp   = imp[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_names)))
    ax.barh(top_names[::-1], top_imp[::-1], color=colors[::-1], edgecolor="k", lw=0.4)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {title}")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


# ── Multi-model comparison plots ─────────────────────────────────────────────

def plot_model_comparison(results_df, save_path=None):
    """Grouped bar: Default RMSE vs Tuned RMSE for every model."""
    df = results_df.copy()
    x  = np.arange(len(df))
    w  = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - w/2, df["Default_RMSE"], w, label="Default",
                   color="#AEC6CF", edgecolor="k", lw=0.5)
    bars2 = ax.bar(x + w/2, df["Tuned_RMSE"],   w, label="Tuned (best params)",
                   color="#4C72B0", edgecolor="k", lw=0.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 80,
                f"${h/1000:.1f}k", ha="center", va="bottom", fontsize=7.5, color="gray")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 80,
                f"${h/1000:.1f}k", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Model Comparison — Default vs Tuned RMSE (lower is better)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cv_scores(results_df, save_path=None):
    """CV RMSE mean ± std bar chart for all models."""
    df = results_df.copy()
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [PALETTE.get(m, DEFAULT_COLOR) for m in df["Model"]]
    ax.bar(x, df["CV_RMSE_Mean"], yerr=df["CV_RMSE_Std"],
           color=colors, edgecolor="k", lw=0.5,
           capsize=5, alpha=0.85, label="CV RMSE (mean ± std)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.set_ylabel("CV RMSE ($)")
    ax.set_title("5-Fold Cross-Validation RMSE (mean ± std)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_r2_comparison(results_df, save_path=None):
    """Horizontal bar chart of R² scores."""
    df = results_df.sort_values("R2")
    colors = [PALETTE.get(m, DEFAULT_COLOR) for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["Model"], df["R2"], color=colors, edgecolor="k", lw=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w - 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.4f}", ha="right", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("R² Score")
    ax.set_title("R² Score Comparison — All Models (higher is better)")
    ax.axvline(1.0, color="red", ls="--", lw=1, alpha=0.5)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tuning_time(results_df, save_path=None):
    """Bar chart: tuning time per model."""
    df = results_df[results_df["Tuning_Time_s"] > 0].copy()
    if df.empty:
        return
    colors = [PALETTE.get(m, DEFAULT_COLOR) for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["Model"], df["Tuning_Time_s"], color=colors, edgecolor="k", lw=0.5)
    ax.set_ylabel("Tuning Time (s)")
    ax.set_title("Hyperparameter Tuning Time per Model")
    ax.tick_params(axis="x", rotation=15)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["Tuning_Time_s"] + 0.5, f"{row['Tuning_Time_s']:.1f}s",
                ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
