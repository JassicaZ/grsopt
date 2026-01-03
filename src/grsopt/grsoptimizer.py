import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from itertools import product
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

class grsoptimizer:
    def __init__(self, original_betas, regularization_strength=0.01, 
                 max_iter=2000, n_intial=5, noise_scale =0.3,):
        """
        GRS Beta值优化器 
        
        Parameters:
        -----------
        original_betas : array-like
            原始GRS模型的beta值
        regularization_strength : float
            正则化强度，小样本建议用更小的值
        max_iter : int
            最大迭代次数
        n_intial : int
            不同初始点的数量，用于跳出局部最优
        """
        self.original_betas = original_betas
        self.regularization_strength = regularization_strength
        self.max_iter = max_iter
        self.n_intial = n_intial
        self.noise_scale = noise_scale
        self.optimized_betas = None
        self.optimization_history = []
        
        
    def _calculate_grs(self, snp_data, betas):
        """计算GRS得分"""
        return np.dot(snp_data, betas)
    
    def _sigmoid(self,z):
        """Sigmoid函数，且防止数值溢出"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
     
    def _objective_function(self, betas, X_train, y_train):
        """
        目标函数：最大化验证集AUC - 正则化项
        """
        try:
            # 计算训练集的GRS
            grs_train = self._calculate_grs(X_train, betas)
            
            # 2. 转换为概率 (Probability)
            probs_train = self._sigmoid(grs_train)
            
            #计算log-loss
            loss = log_loss(y_train, probs_train)#1=disease,0=healthy

            
            # 正则化项：惩罚与原始beta的偏差
            regularization = self.regularization_strength * np.sum((betas - self.original_betas)**2)
            
            # 目标：最大化AUC，最小化正则化项
            objective = loss + regularization
            
            return objective
            
        except Exception as e:
            # 如果出现数值问题，返回较大的目标值
            return 1.0
    
    def _gradient_function(self, betas, X_train, y_train):
        """
        计算目标函数关于 betas 的导数 (梯度)
        """
        N = len(y_train)
        
        grs_train = self._calculate_grs(X_train, betas)
        probs = self._sigmoid(grs_train)
        
        # Log-Loss 的梯度
        grad_loss = np.dot(X_train.T, (probs - y_train)) / N
        # 正则化项的梯度
        grad_reg = 2 * self.regularization_strength * (betas - self.original_betas)
        # 总梯度
        return grad_loss + grad_reg

    
    
    def optimize(self, X, y):
        """
        优化beta值
        
        Parameters:
        -----------
        X : dataframe
            SNP数据       
        y : array-like    
            训练集标签（0/1） 

        Returns:
        --------
        dict : 优化结果
        """
        X = np.array(X.reset_index(drop=True))
        y= np.array(y)


        # 计算原始模型的AUC作为基线
        grs_test = self._calculate_grs(X,self.original_betas)
        
        original_auc = roc_auc_score(y, grs_test)
        
        print(f"原始模型测试集AUC: {original_auc:.4f}")
        
        working_betas = self.original_betas
        #beta优化
        best_local_auc = 0
        best_local_betas = None
        n_intial = self.n_intial
        noise_scale = self.noise_scale
        for seed in range(n_intial):  # 增加到5个不同的初始点
            np.random.seed(seed)
            
            # 更大的随机扰动，帮助跳出局部最优
            initial_betas = working_betas + np.random.normal(0, noise_scale, len(working_betas))
            
            try:
                method = 'L-BFGS-B'
                options = {'maxiter': self.max_iter, 'disp': False}
                
                result = minimize(
                    fun= self._objective_function,
                    x0=initial_betas, 
                    args=(X, y),#计算grs时会挑出筛选的snp，所以这里要全部的
                    jac=self._gradient_function,  # 传入梯度函数
                    method=method,
                   # bounds=bounds, #待加入自定义功能
                    options=options,
                    callback=self._callback_function
                )
                
                # 评估当前结果
                current_betas = result.x
                grs = self._calculate_grs(X, current_betas)
                
                current_auc = roc_auc_score(y, grs)
                
                if current_auc > best_local_auc:
                    best_local_auc = current_auc
                    best_local_betas = current_betas
                    
            except Exception as e:
                print(f"发生异常: {e}")
                raise
        
        print(f"训练集AUC: {best_local_auc:.4f}")
        
        if best_local_auc > original_auc:
            best_auc = best_local_auc
            best_betas = best_local_betas
        else:
            best_auc = original_auc
            best_betas = working_betas

        # 使用最优beta值和筛选后的snp
        self.optimized_betas = best_betas
        orginal_selected_betas=self.original_betas

        # 返回结果
        results = {
            'original_auc': original_auc,
            'optimized_auc': best_auc,
            'auc_improvement': best_auc - original_auc,
            'original_betas': self.original_betas,
            'optimized_betas': self.optimized_betas,
            'beta_changes': self.optimized_betas - orginal_selected_betas
        }
        print(f"\n=== 优化结果 ===")
        print(f"原始模型AUC: {original_auc:.4f}")
        print(f"优化后AUC: {best_auc:.4f}")
        print(f"AUC提升: {best_auc - original_auc:.4f}")
        
        return results
    

    def _callback_function(self, xk):
        """优化过程的回调函数，记录历史"""
        self.optimization_history.append(xk.copy())
    
    def cross_validate(self, X, y, param_grid, cv_folds=5, random_state=42):
        """
        交叉验证评估并筛选最佳超参数

        Parameters:
        -----------
        X : array-like
            SNP数据
        y : array-like
            标签（0/1）
        param_grid : dict
            超参数网格，例如：
            {
                 'regularization_strength': [0.001, 0.01, 0.1],
                 'noise_scale' :[0.1,0.3],
            }
        cv_folds : int
            交叉验证的折数
        random_state : int
            随机种子

        Returns:
        --------
        dict : 包含最佳超参数和对应的验证集AUC
        """


        # 检查param_grid中的超参数是否有效
        valid_params = {'regularization_strength','top_snps_ratio', 'n_intial', 'noise_scale'}
        invalid_params = [key for key in param_grid.keys() if key not in valid_params]
        if invalid_params:
            raise ValueError(f"param_grid包含无效的超参数: {invalid_params}. 可调整的超参数为: {valid_params}")


        # 初始化变量
        best_params = None
        best_auc = -np.inf

        # 生成所有超参数组合
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        print(f"开始{cv_folds}折交叉验证，共有{len(param_combinations)}组超参数组合...")

        for params in param_combinations:
            # 设置当前超参数
            current_params = dict(zip(param_names, params))
            self.regularization_strength = current_params.get('regularization_strength', self.regularization_strength)
            self.n_intial = current_params.get('n_intial', self.n_intial)
            self.noise_scale = current_params.get('noise_scale', self.noise_scale)

            fold_aucs = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.loc[train_idx,:], X.loc[val_idx,:]
                y_train, y_val = y[train_idx], y[val_idx]

                # 优化beta
                self.optimize(X_train, y_train)

                # 在验证集上评估
                grs_val = self._calculate_grs(np.array(X_val), self.optimized_betas)
                val_auc = roc_auc_score(y_val, grs_val)
                fold_aucs.append(val_auc)


            # 计算当前超参数组合的平均AUC
            mean_auc = np.mean(fold_aucs)
            print(f"参数{current_params}的平均验证集AUC: {mean_auc:.4f}")

            # 更新最佳参数
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = current_params
            
        print(f"最佳参数: {json.dumps(best_params, indent=2)}")
        print(f"最佳平均验证集AUC: {best_auc:.4f}")

        return {
            'best_params': best_params,
            'best_auc': best_auc
        }