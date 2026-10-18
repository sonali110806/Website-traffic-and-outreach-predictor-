
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "anomaly_model_improved.joblib"
IMPUTER_PATH = MODEL_DIR / "anomaly_imputer_improved.joblib"

from ml.risk_detection.anomaly_detection.anomaly_model import ANOMALY_FEATURES


# Exploratory threshold from the earlier comparison.
# Validate on a separate validation set before production use.
DEFAULT_SCORE_THRESHOLD = 0.684863


def load_artifacts():
    """Load the saved Isolation Forest model and fitted imputer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained anomaly model not found: {MODEL_PATH}. "
            "Run train_anomaly.py first."
        )

    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(
            f"Saved imputer not found: {IMPUTER_PATH}. "
            "Run train_anomaly.py first."
        )

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer


def detect_anomalies(data, threshold=DEFAULT_SCORE_THRESHOLD):
    """
    Detect unusual telemetry rows.

    Args:
        data: A pandas DataFrame containing ANOMALY_FEATURES.
        threshold: Minimum anomaly score required to flag a row.
                   Higher thresholds generally flag fewer rows.

    Returns:
        A copy of the input DataFrame with anomaly_score and
        is_anomaly columns appended.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing = [col for col in ANOMALY_FEATURES if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    features = data[ANOMALY_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    model, imputer = load_artifacts()
    features_imputed = imputer.transform(features)

    # Negate the score so larger values indicate more unusual rows.
    scores = -model.score_samples(features_imputed)

    result = data.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = (scores >= threshold).astype(int)

    return result


if __name__ == "__main__":
    dataset_path = (
        PROJECT_ROOT
        / "website_outage_dataset"
        / "improved_data"
        / "outage_risk_dataset.csv"
    )
    df = pd.read_csv(dataset_path)

    # Test on the last 10 rows; labels are used only for display.
    result = detect_anomalies(df.tail(10))

    columns_to_show = [
        "timestamp",
        "anomaly_score",
        "is_anomaly",
    ]
    if "anomaly_label" in result.columns:
        columns_to_show.append("anomaly_label")

    print(result[columns_to_show].to_string(index=False))
