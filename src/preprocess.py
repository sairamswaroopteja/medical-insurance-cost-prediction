"""
preprocess.py
-------------
Loads, cleans, encodes, and scales the insurance dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df.drop("region", axis=1), region_dummies], axis=1)
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_charges"] = np.log(df["charges"])
    return df


def scale_features(X_train, X_test, feature_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    return X_train_scaled, X_test_scaled, scaler
