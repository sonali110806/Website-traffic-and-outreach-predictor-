
from pathlib import Path
import sys

import pandas as pd

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.risk_detection.risk_engine.risk_calculator import calculate_risk

DATA_PATH = (
    PROJECT_ROOT
    / "website_outage_dataset"
    / "improved_data"
    / "outage_risk_dataset.csv"
)


def predict_risk(data):
    """Compatibility wrapper around the shared risk engine."""
    return calculate_risk(data)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    healthy = df.loc[
        (df["website_status"] == "healthy")
        & (df["risk_label_valid"] == 1)
    ].tail(10)

    result = predict_risk(healthy)

    columns = [
        "timestamp",
        "risk_probability",
        "risk_threshold",
        "predicted_outage_risk",
    ]

    print("Outage-risk prediction demo")
    print(result[columns].to_string(index=False))
