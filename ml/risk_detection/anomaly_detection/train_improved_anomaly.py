from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (
    PROJECT_ROOT
    / "website_outage_dataset"
    / "improved_data"
    / "outage_risk_dataset.csv"
)
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "anomaly_model_improved.joblib"
IMPUTER_PATH = MODEL_DIR / "anomaly_imputer_improved.joblib"
REPORT_PATH = MODEL_DIR / "anomaly_evaluation_improved.json"

sys.path.insert(0, str(PROJECT_ROOT))

from ml.risk_detection.anomaly_detection.anomaly_model import (
    ANOMALY_FEATURES,
    build_anomaly_model,
)


def choose_threshold(scores, labels):
    """Select the threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(
        labels, scores, pos_label=1
    )

    if len(thresholds) == 0:
        raise ValueError("Cannot select a threshold from validation scores.")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = ANOMALY_FEATURES + ["anomaly_label", "timestamp"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["anomaly_label"] = pd.to_numeric(
        df["anomaly_label"], errors="coerce"
    )
    df = (
        df.dropna(subset=["timestamp", "anomaly_label"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    labels = df["anomaly_label"].astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("anomaly_label must contain only 0 and 1.")

    features = df[ANOMALY_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # Chronological 60/20/20 split with 30 rows purged
    # at each boundary. The dataset is minute-level.
    n = len(df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    gap = 30

    train_idx = np.arange(0, train_end - gap)
    validation_idx = np.arange(train_end, validation_end - gap)
    test_idx = np.arange(validation_end, n)

    X_train = features.iloc[train_idx]
    X_validation = features.iloc[validation_idx]
    X_test = features.iloc[test_idx]

    y_validation = labels.iloc[validation_idx].to_numpy()
    y_test = labels.iloc[test_idx].to_numpy()

    if len(np.unique(labels.iloc[train_idx])) < 1:
        raise ValueError("Training data is empty.")

    if len(np.unique(y_validation)) < 2:
        raise ValueError(
            "Validation labels need both classes to select a useful threshold."
        )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)
    X_test_imputed = imputer.transform(X_test)

    model = build_anomaly_model()
    model.fit(X_train_imputed)

    # Higher scores indicate more anomalous observations.
    validation_scores = -model.score_samples(X_validation_imputed)
    threshold, validation_best_f1 = choose_threshold(
        validation_scores, y_validation
    )

    test_scores = -model.score_samples(X_test_imputed)
    test_predictions = (test_scores >= threshold).astype(int)

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "IsolationForest",
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "evaluation_method": (
            "Chronological 60/20/20 split with 30-minute "
            "purge gaps; threshold selected on validation set"
        ),
        "threshold_selection": "Maximum validation F1",
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "test_rows": int(len(test_idx)),
        "train_anomalies": int(labels.iloc[train_idx].sum()),
        "validation_anomalies": int(y_validation.sum()),
        "test_anomalies": int(y_test.sum()),
        "threshold": threshold,
        "validation_best_f1": validation_best_f1,
        "predicted_test_anomalies": int(test_predictions.sum()),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, test_predictions, zero_division=0)
        ),
        "average_precision_pr_auc": (
            float(average_precision_score(y_test, test_scores))
            if len(np.unique(y_test)) == 2
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y_test, test_scores))
            if len(np.unique(y_test)) == 2
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "note": (
            "Synthetic-data evaluation only. Threshold was selected "
            "on validation data, not test data. Results do not establish "
            "real-world anomaly detection performance."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Improved anomaly model training completed.")
    print(f"Training rows:   {len(train_idx)}")
    print(f"Validation rows: {len(validation_idx)}")
    print(f"Test rows:       {len(test_idx)}")
    print(f"Anomalies train/validation/test: "
          f"{results['train_anomalies']}/"
          f"{results['validation_anomalies']}/"
          f"{results['test_anomalies']}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Validation best F1: {validation_best_f1:.4f}")
    print(f"Test precision: {results['precision']:.4f}")
    print(f"Test recall:    {results['recall']:.4f}")
    print(f"Test F1-score:  {results['f1_score']:.4f}")
    print(f"Test PR-AUC:    {results['average_precision_pr_auc']}")
    print(f"Test ROC-AUC:   {results['roc_auc']}")
    print("Test confusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix"])
    print(f"Saved model:   {MODEL_PATH}")
    print(f"Saved imputer: {IMPUTER_PATH}")
    print(f"Saved report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
