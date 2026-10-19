"""
Member 2: Website traffic forecasting
Target: requests_per_minute 30 minutes into the future.
Run from the repository root:
    python ml/traffic_forecasting/train_forecast_model.py

This script derives its own past-only lag features to avoid relying on
precomputed lag columns that may be inconsistent. It evaluates chronologically.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_CANDIDATES = [
    PROJECT_ROOT / "website_outage_dataset" / "traffic_forecasting_dataset.csv",
    PROJECT_ROOT / "website_outage_dataset" / "improved_data" / "traffic_forecasting_dataset.csv",
    PROJECT_ROOT / "traffic_forecasting_dataset.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = OUTPUT_DIR / "forecast_model.pkl"
REPORT_PATH = OUTPUT_DIR / "forecast_evaluation.json"
GRAPH_PATH = OUTPUT_DIR / "forecast_graph.png"

HORIZON_MINUTES = 30


def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find traffic_forecasting_dataset.csv. Tried:\n" + tried
    )


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    required = ["timestamp", "requests_per_minute"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # The dataset is expected to have one row per minute.
    # Reindexing makes the time interval explicit; missing traffic remains NaN
    # and is removed below rather than silently treated as zero.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    traffic = pd.to_numeric(df["requests_per_minute"], errors="coerce")
    features = pd.DataFrame(index=df.index)

    # Past-only traffic features: at timestamp t, these use data no later than t.
    features["traffic_now"] = traffic
    for lag in [1, 5, 15, 30, 60]:
        features[f"traffic_lag_{lag}m"] = traffic.shift(lag)

    features["traffic_rolling_mean_5m"] = traffic.shift(1).rolling(5).mean()
    features["traffic_rolling_mean_15m"] = traffic.shift(1).rolling(15).mean()
    features["traffic_rolling_std_15m"] = traffic.shift(1).rolling(15).std()

    # Calendar features are known at prediction time.
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["is_weekend"] = (features.index.dayofweek >= 5).astype(int)

    # Current active users can be used if present and available at prediction time.
    if "active_users" in df.columns:
        features["active_users"] = pd.to_numeric(
            df["active_users"], errors="coerce"
        )

    # Direct 30-minute-ahead target.
    target = traffic.shift(-HORIZON_MINUTES).rename("target_traffic_30m_ahead")
    dataset = features.join(target).dropna()

    feature_cols = list(features.columns)
    return dataset, feature_cols
