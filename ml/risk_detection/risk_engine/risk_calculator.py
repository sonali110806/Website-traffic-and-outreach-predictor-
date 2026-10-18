from pathlib import Path
import json

import joblib
import pandas as pd

from ml.risk_detection.outage_prediction.risk_model import RISK_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "risk_model_improved.joblib"
IMPUTER_PATH = MODEL_DIR / "risk_imputer_improved.joblib"
REPORT_PATH = MODEL_DIR / "risk_evaluation_improved.json"


def load_risk_artifacts():
    """Load the improved risk model, imputer, and saved threshold."""
    for path in (MODEL_PATH, IMPUTER_PATH, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact not found: {path}")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    threshold = float(report["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Saved risk threshold must be between 0 and 1.")

    return model, imputer, threshold


def calculate_risk(data):
    """
    Estimate the probability of an incident starting within 30 minutes.

    Args:
        data: DataFrame containing all RISK_FEATURES. If website_status
              is provided, every row must be currently healthy.

    Returns:
        A copy of the input with risk_probability, risk_threshold,
        and predicted_outage_risk columns.

    Note:
        This is an experimental model trained on synthetic data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data must contain at least one row")

    missing = [feature for feature in RISK_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if "website_status" in data.columns:
        statuses = data["website_status"].astype(str).str.lower()
        if not statuses.eq("healthy").all():
            raise ValueError(
                "The risk model is intended for currently healthy rows only."
            )

    features = data[RISK_FEATURES].apply(pd.to_numeric, errors="coerce")

    # Reject rows where every input feature is missing or nonnumeric.
    if features.isna().all(axis=1).any():
        raise ValueError(
            "At least one row has no usable numeric feature values."
        )

    model, imputer, threshold = load_risk_artifacts()
    probabilities = model.predict_proba(imputer.transform(features))[:, 1]

    result = data.copy()
    result["risk_probability"] = probabilities
    result["risk_threshold"] = threshold
    result["predicted_outage_risk"] = (
        probabilities >= threshold
    ).astype(int)

    return result
