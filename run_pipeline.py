"""
run_pipeline.py  (v2 — Production-Level)
-----------------------------------------
End-to-end ML pipeline for Medical Insurance Cost Prediction.

Steps
-----
1.  Load & clean data
2.  EDA (retained from v1)
3.  Preprocessing + Feature Engineering
4.  Train/test split
5.  Train all models with hyperparameter tuning
    Models: LinearRegression, Ridge, Lasso, KNN, RandomForest, AdaBoost
6.  Per-model evaluation plots (pred vs actual, residuals)
7.  Global comparison plots (RMSE, R², CV, tuning time)
8.  Feature importance + SHAP for best model
9.  Save results.csv + all individual models + best_model.pkl
10. Risk segmentation
11. Final summary
"""

import sys, os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, shap

from sklearn.model_selection import train_test_split
from src.train import train_all, compute_metrics, inverse_log 
from src.preprocess        import load_data, encode, add_log_target, scale_features
from src.feature_engineering import add_features
from src.evaluate import (
    plot_pred_vs_actual, plot_residuals, plot_feature_importance,
    plot_model_comparison, plot_cv_scores, plot_r2_comparison,
    plot_tuning_time,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA   = os.path.join(BASE, "insurance.csv")
PLOTS  = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
os.makedirs(PLOTS,  exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & clean
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 1 — Load & Clean")
print("="*65)
df_raw = load_data(DATA)
print(f"  Shape after cleaning : {df_raw.shape}")
print(f"  Missing values       : {df_raw.isnull().sum().sum()}")
print(df_raw.describe().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EDA plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 2 — EDA Plots")
print("="*65)

# 2a. Charges distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df_raw["charges"], bins=50, color="#4C72B0", edgecolor="k", alpha=0.8)
axes[0].set_title("Charges (Raw)"); axes[0].set_xlabel("Charges ($)")
axes[1].hist(np.log(df_raw["charges"]), bins=50, color="#DD8452", edgecolor="k", alpha=0.8)
axes[1].set_title("log(Charges)"); axes[1].set_xlabel("log(Charges)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/01_charges_distribution.png", dpi=150); plt.close()

# 2b. Smoker vs charges
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_raw, x="smoker", y="charges",
            palette={"yes": "#C44E52", "no": "#4C72B0"}, ax=ax)
ax.set_title("Charges by Smoking Status  ← Most Impactful Feature")
plt.tight_layout(); plt.savefig(f"{PLOTS}/02_charges_vs_smoker.png", dpi=150); plt.close()

# 2c. Age vs charges coloured by smoker
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df_raw["age"], df_raw["charges"],
                c=df_raw["smoker"].map({"yes": 1, "no": 0}),
                cmap="coolwarm", alpha=0.5, s=25)
plt.colorbar(sc, ax=ax, label="Smoker (1=yes)")
ax.set_title("Charges vs Age"); ax.set_xlabel("Age"); ax.set_ylabel("Charges ($)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/03_charges_vs_age.png", dpi=150); plt.close()

# 2d. BMI × smoker
fig, ax = plt.subplots(figsize=(8, 5))
for s, col in [("yes", "#C44E52"), ("no", "#4C72B0")]:
    sub = df_raw[df_raw["smoker"] == s]
    ax.scatter(sub["bmi"], sub["charges"], label=f"Smoker={s}",
               alpha=0.4, s=20, color=col)
ax.set_title("BMI vs Charges by Smoker"); ax.set_xlabel("BMI"); ax.set_ylabel("Charges ($)"); ax.legend()
plt.tight_layout(); plt.savefig(f"{PLOTS}/04_bmi_charges_smoker.png", dpi=150); plt.close()

# 2e. Correlation heatmap
df_enc_eda = encode(add_log_target(df_raw))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_enc_eda.select_dtypes(include=np.number).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{PLOTS}/05_correlation_heatmap.png", dpi=150); plt.close()

# 2f. Region / children
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df_raw, x="region",   y="charges", palette="Set2", ax=axes[0])
axes[0].set_title("Charges by Region"); axes[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=df_raw, x="children", y="charges", palette="Set3", ax=axes[1])
axes[1].set_title("Charges by Number of Children")
plt.tight_layout(); plt.savefig(f"{PLOTS}/06_charges_region_children.png", dpi=150); plt.close()

print("  EDA plots saved ✓")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Preprocessing + Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 3 — Preprocessing + Feature Engineering")
print("="*65)
df = encode(add_log_target(df_raw))
df = add_features(df)
print(f"  Total features after engineering : {df.shape[1]}")
print(f"  Feature list : {[c for c in df.columns if c not in ['charges','log_charges']]}")

TARGET_LOG = "log_charges"
TARGET_RAW = "charges"
feature_cols = [c for c in df.columns if c not in [TARGET_LOG, TARGET_RAW]]

X     = df[feature_cols]
y_log = df[TARGET_LOG]
y_raw = df[TARGET_RAW]

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw, test_size=0.2, random_state=42
)

num_cols = ["age", "bmi", "children", "age_squared", "smoker_bmi"]
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, num_cols)
print(f"  Train size : {len(X_train)}  |  Test size : {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train all models with hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 4 — Hyperparameter Tuning + Training")
print("="*65)

results_df, trained_models = train_all(
    X_train_sc, y_train_log,
    X_test_sc,  y_test_log,
    models_dir=MODELS,
    log_target=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save results.csv
# ─────────────────────────────────────────────────────────────────────────────
results_csv = f"{BASE}/results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\n  Results saved → {results_csv}")

print("\n" + "="*65)
print("  FULL MODEL COMPARISON TABLE")
print("="*65)
display_cols = ["Model","Default_RMSE","Tuned_RMSE","Improvement",
                "MAE","R2","CV_RMSE_Mean","CV_RMSE_Std","Tuning_Time_s"]
print(results_df[display_cols].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Per-model evaluation plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 5 — Per-Model Evaluation Plots")
print("="*65)

for i, (name, model) in enumerate(trained_models.items(), start=1):
    preds_log = model.predict(X_test_sc)
    preds     = inverse_log(preds_log)
    actuals   = y_test_raw.values

    safe_name = name.replace(" ", "_")
    plot_pred_vs_actual(
        actuals, preds,
        title=f"{name} — Predicted vs Actual",
        save_path=f"{PLOTS}/model_{safe_name}_pred_vs_actual.png",
    )
    plot_residuals(
        actuals, preds,
        title=f"{name}",
        save_path=f"{PLOTS}/model_{safe_name}_residuals.png",
    )
    print(f"  Plots saved for {name}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Global comparison plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 6 — Global Comparison Plots")
print("="*65)

plot_model_comparison(results_df, save_path=f"{PLOTS}/compare_01_rmse.png")
plot_cv_scores(results_df,        save_path=f"{PLOTS}/compare_02_cv_scores.png")
plot_r2_comparison(results_df,    save_path=f"{PLOTS}/compare_03_r2.png")
plot_tuning_time(results_df,      save_path=f"{PLOTS}/compare_04_tuning_time.png")
print("  Comparison plots saved ✓")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Best model: feature importance + SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 7 — Best Model Analysis")
print("="*65)

best_name  = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_row   = results_df.iloc[0]

print(f"  Best model  : {best_name}")
print(f"  Tuned RMSE  : ${best_row['Tuned_RMSE']:,.2f}")
print(f"  MAE         : ${best_row['MAE']:,.2f}")
print(f"  R²          : {best_row['R2']:.4f}")
print(f"  CV RMSE     : ${best_row['CV_RMSE_Mean']:,.2f} ± ${best_row['CV_RMSE_Std']:,.2f}")
print(f"  Best params : {best_row['Best_Params']}")

plot_feature_importance(
    best_model, list(X.columns), best_name,
    save_path=f"{PLOTS}/best_feature_importance.png",
)

# SHAP
try:
    if hasattr(best_model, "estimators_"):
        # Tree-based ensemble → fast TreeExplainer  (e.g. RandomForest)
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_sc)
    else:
        # Non-tree model (e.g. AdaBoost with stumps, KNN, linear) → KernelExplainer on a sample
        background  = shap.sample(X_test_sc, 50)
        explainer   = shap.KernelExplainer(best_model.predict, background)
        shap_values = explainer.shap_values(X_test_sc[:100])

    fig, _ = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_sc, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/best_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, _ = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test_sc, show=False)
    plt.title(f"SHAP Summary — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/best_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP plots saved ✓")
except Exception as e:
    print(f"  SHAP skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save best_model.pkl (with full metadata)
# ─────────────────────────────────────────────────────────────────────────────
best_payload = {
    "model":        best_model,
    "model_name":   best_name,
    "scaler":       scaler,
    "feature_cols": list(X.columns),
    "num_cols":     num_cols,
    "metrics": {
        "Tuned_RMSE":   best_row["Tuned_RMSE"],
        "MAE":          best_row["MAE"],
        "R2":           best_row["R2"],
        "CV_RMSE_Mean": best_row["CV_RMSE_Mean"],
        "CV_RMSE_Std":  best_row["CV_RMSE_Std"],
    },
    "best_params": best_row["Best_Params"],
}
joblib.dump(best_payload, f"{MODELS}/best_model.pkl")
print(f"  best_model.pkl saved → {MODELS}/best_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Risk segmentation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 8 — Risk Segmentation")
print("="*65)

bins_risk   = [0, 5000, 15000, np.inf]
labels_risk = ["Low", "Medium", "High"]
df["risk_segment"] = pd.cut(df[TARGET_RAW], bins=bins_risk, labels=labels_risk)
seg = df["risk_segment"].value_counts().sort_index()
print(seg.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
seg.plot(kind="bar", color=["#55A868", "#F7C948", "#C44E52"], edgecolor="k", ax=ax)
ax.set_title("Patient Risk Segmentation"); ax.set_xlabel("Risk Group")
ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(f"{PLOTS}/risk_segmentation.png", dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PIPELINE COMPLETE — FINAL SUMMARY")
print("="*65)
print(f"\n  ★  Best model   : {best_name}")
print(f"     Tuned RMSE   : ${best_row['Tuned_RMSE']:>10,.2f}")
print(f"     MAE          : ${best_row['MAE']:>10,.2f}")
print(f"     R²           : {best_row['R2']:.4f}")
print(f"     CV RMSE      : ${best_row['CV_RMSE_Mean']:>10,.2f} ± ${best_row['CV_RMSE_Std']:,.2f}")

print("\n  Output files")
print(f"  ├── results.csv                  → full model comparison")
print(f"  ├── models/best_model.pkl        → best model + scaler + metadata")
for name in trained_models:
    print(f"  ├── models/{name}.pkl")
print(f"  └── plots/                       → {len(list(__import__('os').listdir(PLOTS)))} plots")

print("\n  All model results (sorted by Tuned RMSE):")
print(results_df[["Model","Tuned_RMSE","MAE","R2","CV_RMSE_Mean"]].to_string(index=False))
print("\n" + "="*65)
