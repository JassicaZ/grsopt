import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import grsopt

class TestGRSBetaOptimizer(unittest.TestCase):

    def setUp(self):
        # 初始化测试数据

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
        # 测试 optimize 方法是否可以运行
        results = self.optimizer.optimize(self.X, self.y)
        self.assertIsInstance(results, dict)  # 检查返回值是否为字典
        self.assertIn('optimized_auc', results)  # 检查返回值是否包含关键字段
        self.assertIn('optimized_betas', results)

    def test_calculate_grs(self):
        # 测试 _calculate_grs 方法是否可以运行
        grs = self.optimizer._calculate_grs(self.X, self.original_betas)
        self.assertIsInstance(grs, np.ndarray)  # 检查返回值是否为 NumPy 数组

    def test_sigmoid(self):
        # 测试 _sigmoid 方法是否可以运行
        z = np.array([-1000, 0, 1000])
        sigmoid_values = self.optimizer._sigmoid(z)
        self.assertIsInstance(sigmoid_values, np.ndarray)  # 检查返回值是否为 NumPy 数组

    def test_cross_validate(self):
        # 测试 cross_validate 方法是否可以运行
        param_grid = {
             'regularization_strength': [0.001, 0.01],
             'noise_scale' :[0.1,0.3]
        }
        results = self.optimizer.cross_validate(self.X, self.y, param_grid, cv_folds=3)
        self.assertIsInstance(results, dict)  # 检查返回值是否为字典
        self.assertIn('best_params', results)  # 检查返回值是否包含关键字段
        self.assertIn('best_auc', results)

if __name__ == '__main__':
    unittest.main()