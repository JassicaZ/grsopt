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

    def test_cross_validate_rank_loss(self):
        # Verify cross_validate accepts a rank loss and restores it afterwards.
        param_grid = {'regularization_strength': [0.001, 0.01]}
        self.optimizer.loss = 'log'
        results = self.optimizer.cross_validate(
            self.X, self.y, param_grid, cv_folds=3, loss='rank'
        )
        self.assertIsInstance(results, dict)
        self.assertIn('best_params', results)
        self.assertEqual(self.optimizer.loss, 'log')  # restored

    def test_loss_invalid(self):
        # Verify that an unknown loss raises a clear error.
        with self.assertRaises(ValueError):
            grsopt.grsoptimizer(original_betas=self.original_betas, loss='foo')

    def test_optimize_rank_loss(self):
        # Verify that rank loss optimisation runs and returns a result dict.
        optimizer = grsopt.grsoptimizer(
            original_betas=self.original_betas,
            regularization_strength=0.01,
            max_iter=100,
            loss='rank',
        )
        results = optimizer.optimize(self.X, self.y)
        self.assertIsInstance(results, dict)
        self.assertIn('optimized_betas', results)

    def test_rank_loss_gradient(self):
        # Verify the pairwise rank loss gradient against finite differences.
        np.random.seed(0)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        if np.unique(y).size < 2:
            y[0] = 1 - y[0]
        opt = grsopt.grsoptimizer(original_betas=np.random.randn(5), loss='rank')
        betas = np.random.randn(5)

        def loss_fn(b):
            return opt._rank_auc_loss(b, X, y)

        grad = opt._rank_auc_gradient(betas, X, y)
        eps = 1e-6
        num_grad = np.zeros_like(betas)
        for j in range(len(betas)):
            bp = betas.copy(); bp[j] += eps
            bm = betas.copy(); bm[j] -= eps
            num_grad[j] = (loss_fn(bp) - loss_fn(bm)) / (2 * eps)
        np.testing.assert_allclose(grad, num_grad, atol=1e-6)


if __name__ == '__main__':
    unittest.main()