import unittest

import pandas as pd

from main import pre_treatment_features


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


if __name__ == "__main__":
    unittest.main()
