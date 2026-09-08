import unittest

import numpy as np
import pandas as pd

import grsopt


class TestGRSPruner(unittest.TestCase):

    def setUp(self):
        # Initialize a small deterministic dataset for pruning tests.
        np.random.seed(42)
        n_samples = 200
        self.X = pd.DataFrame(
            {
                "SNP_1": np.random.randn(n_samples),
                "SNP_2": np.random.randn(n_samples),
                "SNP_3": np.random.randn(n_samples),
                "SNP_4": np.random.randn(n_samples),
                "SNP_5": np.random.randn(n_samples),
            }
        )
        latent_score = self.X["SNP_1"] + 0.75 * self.X["SNP_2"]
        self.y = (latent_score > np.median(latent_score)).astype(int)

    def test_calculate_auc_all_features(self):
        # Verify that AUC calculation works with all columns selected.
        auc_value = grsopt.calculate_auc(self.X, self.y, selected_features='all')
        self.assertIsInstance(auc_value, float)
        self.assertGreaterEqual(auc_value, 0.0)
        self.assertLessEqual(auc_value, 1.0)

    def test_calculate_auc_selected_features(self):
        # Verify that AUC calculation works with a subset of loci.
        selected_features = ["SNP_1", "SNP_2", "SNP_3"]
        auc_value = grsopt.calculate_auc(self.X, self.y, selected_features=selected_features)
        self.assertIsInstance(auc_value, float)
        self.assertGreaterEqual(auc_value, 0.0)
        self.assertLessEqual(auc_value, 1.0)

if __name__ == '__main__':
    unittest.main()
