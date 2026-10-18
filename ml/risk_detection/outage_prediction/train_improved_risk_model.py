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
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = (
    PROJECT_ROOT
    / "website_outage_dataset"
    / "improved_data"
    / "outage_risk_dataset.csv"
)
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "risk_model_improved.joblib"
IMPUTER_PATH = MODEL_DIR / "risk_imputer_improved.joblib"
REPORT_PATH = MODEL_DIR / "risk_evaluation_improved.json"


from ml.risk_detection.outage_prediction.risk_model import (
    RISK_FEATURES,
    build_risk_model,
)


def merge_short_incident_gaps(incident, max_gap=3):
    """Merge healthy gaps of at most max_gap minutes between incidents."""
    values = incident.to_numpy(dtype=bool)
    result = values.copy()
    n = len(values)
    i = 0

    while i < n:
        if values[i]:
            i += 1
            continue

        start = i
        while i < n and not values[i]:
            i += 1
        end = i

        gap_length = end - start
        has_incident_before = start > 0 and values[start - 1]
        has_incident_after = end < n and values[end]

        if (
            gap_length <= max_gap
            and has_incident_before
            and has_incident_after
        ):
            result[start:end] = True

    return pd.Series(result, index=incident.index)


def calculate_corrected_target(df):
    """Label healthy rows if an incident starts in the next 30 minutes."""
    incident = df["website_status"].isin(["degraded", "outage"])
    incident = merge_short_incident_gaps(incident, max_gap=3)

    # An incident starts at the first incident minute after a healthy period.
    starts = incident & ~incident.shift(1, fill_value=False)
    start_values = starts.to_numpy(dtype=bool)

    target = np.zeros(len(df), dtype=np.int8)

    # Exclude the current minute; include the next 30 minutes.
    for i in range(len(df)):
        end = min(len(df), i + 31)
        if i + 1 < end:
            target[i] = int(start_values[i + 1:end].any())

    return incident, starts, target


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = (
        RISK_FEATURES
        + ["timestamp", "website_status", "risk_label_valid"]
    )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    incident, incident_starts, target = calculate_corrected_target(df)
    df["corrected_outage_risk"] = target
    df["is_incident"] = incident

    # Use only rows with complete future-label coverage and a healthy
    # current website state. Do not use the original target as the label.
    eligible = df.loc[
        (df["risk_label_valid"] == 1) & (~df["is_incident"])
    ].copy()
    eligible = eligible.sort_values("timestamp").reset_index(drop=True)

    X = eligible[RISK_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = eligible["corrected_outage_risk"].astype(int)

    if len(eligible) < 100:
        raise ValueError("Too few eligible rows to train the model.")


    # Chronological 60/20/20 split with a 30-minute purge gap.
    n = len(eligible)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_part = eligible.iloc[:train_end].copy()
    val_part = eligible.iloc[train_end:validation_end].copy()
    test_part = eligible.iloc[validation_end:].copy()

    if train_part.empty or val_part.empty or test_part.empty:
        raise ValueError("A chronological data partition is empty.")

    # Preserve the original partition boundaries before purging.
    validation_boundary = val_part["timestamp"].iloc[0]
    test_boundary = test_part["timestamp"].iloc[0]

    purge_duration = pd.Timedelta(minutes=30)

    # Training labels must not depend on incidents in validation.
    train_part = train_part.loc[
        train_part["timestamp"] < validation_boundary - purge_duration
    ].copy()

    # Validation labels must not depend on incidents in test.
    val_part = val_part.loc[
        val_part["timestamp"] < test_boundary - purge_duration
    ].copy()

    # Keep the test partition unchanged.
    X_train = train_part[RISK_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_val = val_part[RISK_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test_part[RISK_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    y_train = train_part["corrected_outage_risk"].astype(int)
    y_val = val_part["corrected_outage_risk"].astype(int)
    y_test = test_part["corrected_outage_risk"].astype(int)


    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0:
        raise ValueError("Training period has no positive labels.")

    model = build_risk_model(
        scale_pos_weight=negatives / positives
    )
    model.fit(X_train_imp, y_train)

    # Select threshold using validation data only.
    val_prob = model.predict_proba(X_val_imp)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob)

    if len(thresholds) == 0:
        raise ValueError("Unable to choose a threshold from validation data.")

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    # Evaluate only once on the untouched chronological test period.
    test_prob = model.predict_proba(X_test_imp)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    report = classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["No incident start", "Incident starts within 30 minutes"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "XGBClassifier",
        "target": (
            "Incident start within the next 30 minutes, "
            "restricted to currently healthy rows"
        ),
        "incident_gap_merge_minutes": 3,
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "evaluation_method": (
        "chronological 60/20/20 split with a 30-minute purge "
        "before validation and test boundaries"
        ),
        "purge_gap_minutes": 30,
        "threshold_selected_on": "validation set only, maximizing F1",
        "threshold": threshold,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_positive_labels": int(y_train.sum()),
        "validation_positive_labels": int(y_val.sum()),
        "test_positive_labels": int(y_test.sum()),
     "train_start": str(train_part["timestamp"].iloc[0]),
         "train_end": str(train_part["timestamp"].iloc[-1]),
         "validation_start": str(val_part["timestamp"].iloc[0]),
         "validation_end": str(val_part["timestamp"].iloc[-1]),
         "test_start": str(test_part["timestamp"].iloc[0]),
        "test_end": str(test_part["timestamp"].iloc[-1]),
        "incident_start_count": int(incident_starts.sum()),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, test_pred, zero_division=0)),
        "average_precision_pr_auc": float(
            average_precision_score(y_test, test_prob)
        ),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "confusion_matrix": confusion_matrix(
            y_test, test_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "note": (
            "Experimental model trained on synthetic data. "
            "Short incident gaps up to three minutes are merged. "
            "Few incident periods limit the reliability of performance estimates."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Corrected outage-risk model training completed.")
    print(f"Training rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows:       {len(X_test)}")
    print(f"Positive labels (train/validation/test): "
          f"{int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Test precision: {results['precision']:.4f}")
    print(f"Test recall:    {results['recall']:.4f}")
    print(f"Test F1-score:  {results['f1_score']:.4f}")
    print(f"Test PR-AUC:    {results['average_precision_pr_auc']:.4f}")
    print(f"Test ROC-AUC:   {results['roc_auc']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix"])
    print(f"\nSaved model:   {MODEL_PATH}")
    print(f"Saved imputer: {IMPUTER_PATH}")
    print(f"Saved report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
