"""
feature_engineering.py
-----------------------
Creates derived features that boost model performance.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Polynomial
    df["age_squared"] = df["age"] ** 2

    # BMI category
    bins = [0, 18.5, 25, 30, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bins, labels=labels)
    bmi_dummies = pd.get_dummies(df["bmi_category"], prefix="bmi_cat", drop_first=True)
    df = pd.concat([df.drop("bmi_category", axis=1), bmi_dummies], axis=1)

    # Has children flag
    df["has_children"] = (df["children"] > 0).astype(int)

    # Interaction: smoker * bmi  (critical feature per EDA)
    smoker_col = df["smoker"] if "smoker" in df.columns else df.get("smoker", 0)
    df["smoker_bmi"] = df["smoker"] * df["bmi"]

    return df
