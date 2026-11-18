import unittest

import pandas as pd

from main import binary_values, pre_treatment_features, treatment_effect_summary, wilson_interval


class FeatureLeakageTests(unittest.TestCase):
    def test_removes_assignment_and_post_campaign_columns(self):
        data = pd.DataFrame({
            "age": [30],
            "region": ["north"],
            "treatment": [1],
            "conversion": [1],
            "segment": ["Mens E-Mail"],
            "visit": [1],
            "spend": [20.0],
        })
        features = pre_treatment_features(data, "treatment", "conversion")
        self.assertEqual(list(features.columns), ["age", "region"])

    def test_binary_inputs_reject_non_binary_values(self):
        with self.assertRaises(ValueError):
            binary_values(pd.Series([0, 0.5, 1]), "treatment")
        with self.assertRaises(ValueError):
            binary_values(pd.Series([0, 0]), "outcome")

    def test_newcombe_interval_uses_wilson_limits_and_counts(self):
        result = treatment_effect_summary(
            outcomes=pd.Series([1, 0, 0, 0]).to_numpy(),
            treatment=pd.Series([1, 1, 0, 0]).to_numpy(),
        )
        self.assertEqual(result["treated_conversions"], 1)
        self.assertEqual(result["control_conversions"], 0)
        self.assertEqual(result["estimate"], 0.5)
        self.assertLess(result["ci95"][0], result["estimate"])
        self.assertGreater(result["ci95"][1], result["estimate"])
        self.assertEqual(wilson_interval(0, 0), None)


if __name__ == "__main__":
    unittest.main()

