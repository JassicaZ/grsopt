import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import grsopt


class TestGRSBetaOptimizer(unittest.TestCase):

    def setUp(self):
        # Initialize the test dataset.
        self.X = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f"SNP_{i}" for i in range(10)]
        )
        self.original_betas = np.random.randn(10)
        self.y = np.random.randint(0, 2, 100)
        self.optimizer = grsopt.grsoptimizer(
            original_betas=self.original_betas,
            regularization_strength=0.01,
            max_iter=100,
        )

    def test_optimize(self):
        # Verify that the optimize method can run successfully.
        results = self.optimizer.optimize(self.X, self.y)
        self.assertIsInstance(results, dict)  
        self.assertIn('optimized_auc', results) 
        self.assertIn('optimized_betas', results)

    def test_calculate_grs(self):
        # Verify that the _calculate_grs method can run successfully.
        grs = self.optimizer._calculate_grs(self.X, self.original_betas)
        self.assertIsInstance(grs, np.ndarray) 

    def test_sigmoid(self):
        # Verify that the _sigmoid method can run successfully.
        z = np.array([-1000, 0, 1000])
        sigmoid_values = self.optimizer._sigmoid(z)
        self.assertIsInstance(sigmoid_values, np.ndarray)  

    def test_cross_validate(self):
        # Verify that the cross_validate method can run successfully.
        param_grid = {
             'regularization_strength': [0.001, 0.01],
             'noise_scale': [0.1, 0.3]
        }
        results = self.optimizer.cross_validate(self.X, self.y, param_grid, cv_folds=3)
        self.assertIsInstance(results, dict) 
        self.assertIn('best_params', results)  


if __name__ == '__main__':
    unittest.main()