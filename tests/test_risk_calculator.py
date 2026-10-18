
import unittest

import pandas as pd

from ml.risk_detection.outage_prediction.risk_model import RISK_FEATURES
from ml.risk_detection.risk_engine.risk_calculator import calculate_risk


class TestRiskCalculator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_csv(
            "website_outage_dataset/improved_data/outage_risk_dataset.csv"
        )

        cls.healthy_data = cls.dataset.loc[
            (cls.dataset["website_status"] == "healthy")
            & (cls.dataset["risk_label_valid"] == 1)
        ].tail(5).copy()

    def test_valid_input_returns_predictions(self):
        result = calculate_risk(self.healthy_data)

        self.assertEqual(len(result), len(self.healthy_data))
        self.assertIn("risk_probability", result.columns)
        self.assertIn("risk_threshold", result.columns)
        self.assertIn("predicted_outage_risk", result.columns)

        self.assertTrue(
            result["risk_probability"].between(0, 1).all()
        )
        self.assertTrue(
            result["predicted_outage_risk"].isin([0, 1]).all()
        )

    def test_missing_feature_is_rejected(self):
        incomplete = self.healthy_data.drop(
            columns=[RISK_FEATURES[0]]
        )

        with self.assertRaises(ValueError):
            calculate_risk(incomplete)

    def test_non_healthy_rows_are_rejected(self):
        unhealthy = self.healthy_data.copy()
        unhealthy.loc[unhealthy.index[0], "website_status"] = "outage"

        with self.assertRaises(ValueError):
            calculate_risk(unhealthy)

    def test_empty_input_is_rejected(self):
        empty_data = self.healthy_data.iloc[0:0].copy()

        with self.assertRaises(ValueError):
            calculate_risk(empty_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
