import unittest

import pandas as pd

from ml.risk_detection.anomaly_detection.anomaly_model import (
    ANOMALY_FEATURES,
)
from ml.risk_detection.anomaly_detection.detect_anomaly import (
    DEFAULT_SCORE_THRESHOLD,
    detect_anomalies,
)


class TestDetectAnomalies(unittest.TestCase):
    def setUp(self):
        # Typical telemetry-shaped input for interface testing.
        self.data = pd.DataFrame(
            [{feature: 1.0 for feature in ANOMALY_FEATURES}]
        )

    def test_valid_input_returns_predictions(self):
        result = detect_anomalies(self.data)

        self.assertIn("anomaly_score", result.columns)
        self.assertIn("is_anomaly", result.columns)
        self.assertEqual(len(result), 1)
        self.assertTrue(
            result["is_anomaly"].isin([0, 1]).all()
        )
        self.assertTrue(
            result["anomaly_score"].notna().all()
        )

    def test_missing_feature_is_rejected(self):
        incomplete = self.data.drop(
            columns=[ANOMALY_FEATURES[0]]
        )

        with self.assertRaisesRegex(ValueError, "Missing required"):
            detect_anomalies(incomplete)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            detect_anomalies(self.data, threshold=1.5)

        with self.assertRaisesRegex(ValueError, "threshold"):
            detect_anomalies(self.data, threshold=-0.1)

    def test_non_dataframe_input_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "DataFrame"):
            detect_anomalies([1, 2, 3])

    def test_default_threshold_matches_improved_model(self):
        self.assertAlmostEqual(
            DEFAULT_SCORE_THRESHOLD, 0.684863, places=6
        )


if __name__ == "__main__":
    unittest.main()
