import unittest

import numpy as np
import pandas as pd

import grsopt
from grsopt.metrics import as_beta_series


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Initialize a small deterministic dataset for metrics tests.
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
        betas = pd.Series(1.0, index=self.X.columns)
        auc_value = grsopt.calculate_auc(self.X, self.y, betas=betas, selected_features='all')
        self.assertIsInstance(auc_value, float)
        self.assertGreaterEqual(auc_value, 0.0)
        self.assertLessEqual(auc_value, 1.0)

    def test_calculate_auc_selected_features(self):
        # Verify that AUC calculation works with a subset of loci.
        selected_features = ["SNP_1", "SNP_2", "SNP_3"]
        betas = pd.Series(1.0, index=self.X.columns)
        auc_value = grsopt.calculate_auc(self.X, self.y, betas=betas, selected_features=selected_features)
        self.assertIsInstance(auc_value, float)
        self.assertGreaterEqual(auc_value, 0.0)
        self.assertLessEqual(auc_value, 1.0)

    def test_calculate_auc_requires_betas(self):
        # Betas are now required; omitting them must raise a ValueError.
        with self.assertRaises(ValueError):
            grsopt.calculate_auc(self.X, self.y, selected_features='all')

    def test_as_beta_series_accepts_series(self):
        s = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
        out = as_beta_series(s, ['A', 'B', 'C'])
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(out.index.tolist(), ['A', 'B', 'C'])
        self.assertEqual(out.tolist(), [1.0, 2.0, 3.0])

    def test_as_beta_series_accepts_dataframe(self):
        df = pd.DataFrame({'b': [1.0, 2.0, 3.0]}, index=['A', 'B', 'C'])
        out = as_beta_series(df, ['A', 'B', 'C'])
        self.assertEqual(out.tolist(), [1.0, 2.0, 3.0])

    def test_as_beta_series_requires_betas(self):
        with self.assertRaises(ValueError):
            as_beta_series(None, ['A', 'B'])

if __name__ == '__main__':
    unittest.main()
