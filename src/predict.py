"""
predict.py
----------
Prediction pipeline for Medical Insurance Cost Prediction.

Usage
-----
  # As a module:
  from predict import predict_insurance_cost
  result = predict_insurance_cost({"age": 30, "sex": "male", ...})

  # As a script:
  python predict.py
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_SRC_DIR   = Path(__file__).resolve().parent   
_PROJ_DIR  = _SRC_DIR.parent        
MODEL_PATH = _PROJ_DIR / "models" / "best_model.pkl"

# ── Valid input values ────────────────────────────────────────────────────────
VALID_SEX    = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGION = {"northeast", "northwest", "southeast", "southwest"}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def _validate(input_dict: dict) -> None:
    """Raise ValueError with a clear message if any field is invalid."""
    required = {"age", "sex", "bmi", "children", "smoker", "region"}
    missing  = required - set(input_dict.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    age = input_dict["age"]
    if not isinstance(age, (int, float)) or not (0 < age < 120):
        raise ValueError(f"'age' must be a number between 1 and 119. Got: {age!r}")

    bmi = input_dict["bmi"]
    if not isinstance(bmi, (int, float)) or not (10 <= bmi <= 70):
        raise ValueError(f"'bmi' must be a number between 10 and 70. Got: {bmi!r}")

    children = input_dict["children"]
    if not isinstance(children, (int, float)) or int(children) < 0:
        raise ValueError(f"'children' must be a non-negative integer. Got: {children!r}")

    sex = str(input_dict["sex"]).lower().strip()
    if sex not in VALID_SEX:
        raise ValueError(f"'sex' must be one of {VALID_SEX}. Got: {sex!r}")

    smoker = str(input_dict["smoker"]).lower().strip()
    if smoker not in VALID_SMOKER:
        raise ValueError(f"'smoker' must be one of {VALID_SMOKER}. Got: {smoker!r}")

    region = str(input_dict["region"]).lower().strip()
    if region not in VALID_REGION:
        raise ValueError(f"'region' must be one of {VALID_REGION}. Got: {region!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (mirrors preprocess.py + feature_engineering.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess(input_dict: dict) -> pd.DataFrame:
    """
    Convert raw input dict → encoded + feature-engineered DataFrame.
    Column order matches training data exactly.
    """
    # 1. Normalise strings
    sex    = str(input_dict["sex"]).lower().strip()
    smoker = str(input_dict["smoker"]).lower().strip()
    region = str(input_dict["region"]).lower().strip()

    # 2. Build base row
    row = {
        "age":      float(input_dict["age"]),
        "sex":      1 if sex == "male" else 0,
        "bmi":      float(input_dict["bmi"]),
        "children": int(input_dict["children"]),
        "smoker":   1 if smoker == "yes" else 0,
        "region":   region,
    }
    df = pd.DataFrame([row])

    # 3. One-hot encode region (drop_first=True → baseline = northeast)
    #    Training used: region_northwest, region_southeast, region_southwest
    for col in ["region_northwest", "region_southeast", "region_southwest"]:
        suffix = col.replace("region_", "")
        df[col] = int(region == suffix)
    df = df.drop("region", axis=1)

    # 4. Feature engineering (mirrors feature_engineering.py)
    df["age_squared"] = df["age"] ** 2

    bmi_val = float(input_dict["bmi"])
    df["bmi_cat_normal"]    = int(18.5 <= bmi_val < 25)
    df["bmi_cat_overweight"] = int(25   <= bmi_val < 30)
    df["bmi_cat_obese"]      = int(bmi_val >= 30)

    df["has_children"] = int(int(input_dict["children"]) > 0)
    df["smoker_bmi"]   = df["smoker"] * df["bmi"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────────────────────
def predict_insurance_cost(input_dict: dict, model_path: str = None) -> dict:
    """
    Predict medical insurance cost for a single individual.

    Parameters
    ----------
    input_dict : dict
        {
          "age": 30,        # int/float, 1–119
          "sex": "male",    # "male" | "female"
          "bmi": 27.5,      # float, 10–70
          "children": 1,    # int >= 0
          "smoker": "no",   # "yes" | "no"
          "region": "northwest"  # "northeast"|"northwest"|"southeast"|"southwest"
        }
    model_path : str, optional
        Path to best_model.pkl. Defaults to models/best_model.pkl relative
        to this file.

    Returns
    -------
    dict
        {
          "predicted_cost": 6425.82,   # dollars
          "model_used":     "GradientBoosting",
          "model_r2":       0.9017,
          "model_rmse":     4250.24,
          "input_summary":  { ... normalised input ... }
        }
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    _validate(input_dict)

    # ── Load model artifact ───────────────────────────────────────────────────
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {path}\n"
            "Run run_pipeline.py first to train and save the model."
        )
    payload = joblib.load(path)

    model        = payload["model"]
    scaler       = payload["scaler"]
    feature_cols = payload["feature_cols"]  # exact training column order
    num_cols     = payload["num_cols"]       # columns to scale
    model_name   = payload["model_name"]
    metrics      = payload["metrics"]

    # ── Build feature matrix ──────────────────────────────────────────────────
    df = _preprocess(input_dict)

    # Align columns to training order (fills any missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]   # enforce exact column order

    # ── Scale numeric features ────────────────────────────────────────────────
    df[num_cols] = scaler.transform(df[num_cols])

    # ── Predict ───────────────────────────────────────────────────────────────
    log_prediction = model.predict(df)[0]
    predicted_cost = float(np.exp(log_prediction))  # inverse log transform

    # ── Build result ──────────────────────────────────────────────────────────
    result = {
        "predicted_cost": round(predicted_cost, 2),
        "model_used":     model_name,
        "model_r2":       float(metrics["R2"]),
        "model_rmse":     float(metrics["Tuned_RMSE"]),
        "input_summary": {
            "age":      int(input_dict["age"]),
            "sex":      str(input_dict["sex"]).lower(),
            "bmi":      float(input_dict["bmi"]),
            "children": int(input_dict["children"]),
            "smoker":   str(input_dict["smoker"]).lower(),
            "region":   str(input_dict["region"]).lower(),
        },
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point — demo predictions
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  Medical Insurance Cost Prediction")
    print("=" * 60)

    try:
        user_input = {
            "age": int(input("Enter age: ")),
            "sex": input("Enter sex (male/female): "),
            "bmi": float(input("Enter BMI: ")),
            "children": int(input("Enter number of children: ")),
            "smoker": input("Smoker? (yes/no): "),
            "region": input("Enter region (northeast/northwest/southeast/southwest): ")
        }

        result = predict_insurance_cost(user_input)

        print("\n Predicted Insurance Cost: $", result["predicted_cost"])

    except Exception as e:
        print("\n Error:", e)