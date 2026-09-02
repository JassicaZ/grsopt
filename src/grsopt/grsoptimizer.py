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
        Optimizer for Genetic Risk Score (GRS) beta coefficients.

        Parameters
        ----------
        original_betas : array-like
            Baseline beta coefficients from the original GRS model.
        regularization_strength : float
            L2 regularization strength applied to deviations from the
            original betas. Use smaller values for small datasets.
        max_iter : int
            Maximum number of optimizer iterations.
        n_intial : int
            Number of distinct random initializations to attempt to help
            escape local minima.
        noise_scale : float
            Standard deviation for random perturbations used to create
            different initial points.
        """
        self.original_betas = original_betas
        self.regularization_strength = regularization_strength
        self.max_iter = max_iter
        self.n_intial = n_intial
        self.noise_scale = noise_scale
        self.optimized_betas = None
        self.optimization_history = []
        
        
    def _calculate_grs(self, snp_data, betas):
        """Compute Genetic Risk Score (GRS) as a dot product.

        Parameters
        ----------
        snp_data : array-like
            Matrix of SNP values (individuals x loci).
        betas : array-like
            Beta coefficients for loci.

        Returns
        -------
        ndarray
            GRS values for each individual.
        """
        return np.dot(snp_data, betas)
    
    def _sigmoid(self,z):
        """Numerically-stable sigmoid / logistic function.

        Clips input to avoid overflow in the exponential.
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
     
    def _objective_function(self, betas, X_train, y_train):
        """
        Objective function for optimization: log-loss with L2 regularization
        that penalizes deviation from the original beta values.

        This function returns a scalar loss to be minimized by the optimizer.
        """
        try:
            grs_train = self._calculate_grs(X_train, betas)
            
            probs_train = self._sigmoid(grs_train)
            loss = log_loss(y_train, probs_train)  # 1 = disease, 0 = healthy

            # L2 regularization penalizing deviation from original betas.
            regularization = self.regularization_strength * np.sum((betas - self.original_betas)**2)

            # Final scalar objective to minimize: data loss + regularization.
            objective = loss + regularization

            return objective
            
        except Exception:
            # On numerical failure, return a large objective value.
            return 1.0
    
    def _gradient_function(self, betas, X_train, y_train):
        """
        Compute gradient of the objective with respect to `betas`.

        Combines the gradient of log-loss and the gradient of the L2 regularizer.
        """
        N = len(y_train)
        
        grs_train = self._calculate_grs(X_train, betas)
        probs = self._sigmoid(grs_train)
        
        # Gradient of log-loss and of the L2 regularizer, summed together.
        grad_loss = np.dot(X_train.T, (probs - y_train)) / N
        grad_reg = 2 * self.regularization_strength * (betas - self.original_betas)
        return grad_loss + grad_reg

    
    
    def optimize(self, X, y):
        """
        Optimize beta coefficients using gradient-based minimization.

        Parameters
        ----------
        X : DataFrame
            SNP data (individuals x loci). Rows correspond to samples.
        y : array-like
            Binary labels (0/1) aligned with X rows.

        Returns
        -------
        dict
            Dictionary containing original and optimized AUCs, betas,
            and the beta differences.
        """
        X = np.array(X.reset_index(drop=True))
        y= np.array(y)


        # Compute baseline AUC using the original betas.
        grs_test = self._calculate_grs(X,self.original_betas)
        
        original_auc = roc_auc_score(y, grs_test)
        
        print(f"Baseline test AUC (original betas): {original_auc:.4f}")
        
        working_betas = self.original_betas
        # Local search for improved betas
        best_local_auc = 0
        best_local_betas = None
        n_intial = self.n_intial
        noise_scale = self.noise_scale
        for seed in range(n_intial):
            np.random.seed(seed)
            
            # Perturb the current betas to produce diverse initializations.
            initial_betas = working_betas + np.random.normal(0, noise_scale, len(working_betas))
            
            try:
                method = 'L-BFGS-B'
                options = {'maxiter': self.max_iter, 'disp': False}
                
                result = minimize(
                    fun= self._objective_function,
                    x0=initial_betas, 
                    args=(X, y),
                    jac=self._gradient_function,
                    method=method,
                    options=options,
                    callback=self._callback_function
                )
                
                # Evaluate the optimization result
                current_betas = result.x
                grs = self._calculate_grs(X, current_betas)
                
                current_auc = roc_auc_score(y, grs)
                
                if current_auc > best_local_auc:
                    best_local_auc = current_auc
                    best_local_betas = current_betas
                    
            except Exception as e:
                print(f"Exception during local optimization: {e}")
                raise
        
        print(f"Best training AUC found (local search): {best_local_auc:.4f}")
        
        if best_local_auc > original_auc:
            best_auc = best_local_auc
            best_betas = best_local_betas
        else:
            best_auc = original_auc
            best_betas = working_betas

        # Store optimized betas (or fall back to original betas).
        self.optimized_betas = best_betas
        orginal_selected_betas=self.original_betas

        # Return a summary of the optimization results.
        results = {
            'original_auc': original_auc,
            'optimized_auc': best_auc,
            'auc_improvement': best_auc - original_auc,
            'original_betas': self.original_betas,
            'optimized_betas': self.optimized_betas,
            'beta_changes': self.optimized_betas - orginal_selected_betas
        }
        print(f"\n=== Optimization Results ===")
        print(f"Original model AUC: {original_auc:.4f}")
        print(f"Optimized AUC: {best_auc:.4f}")
        print(f"AUC improvement: {best_auc - original_auc:.4f}")
        
        return results
    

    def _callback_function(self, xk):
        """Callback for the optimizer: record intermediate betas.

        The optimizer calls this with the current parameter vector at each
        iteration; we store a copy into the optimization history.
        """
        self.optimization_history.append(xk.copy())
    
    def cross_validate(self, X, y, param_grid, cv_folds=5, random_state=42):
        """
        Perform cross-validation over a parameter grid to select hyperparameters.

        Parameters
        ----------
        X : array-like
            SNP data (DataFrame expected by the optimizer).
        y : array-like
            Binary labels (0/1).
        param_grid : dict
            Grid of hyperparameters to evaluate, e.g.::

                {
                    'regularization_strength': [0.001, 0.01, 0.1],
                    'noise_scale': [0.1, 0.3],
                }

        cv_folds : int
            Number of cross-validation folds.
        random_state : int
            Random seed for fold splitting.

        Returns
        -------
        dict
            Dictionary containing `best_params` and `best_auc`.
        """


        # Validate parameter names in the provided grid.
        valid_params = {'regularization_strength','top_snps_ratio', 'n_intial', 'noise_scale'}
        invalid_params = [key for key in param_grid.keys() if key not in valid_params]
        if invalid_params:
            raise ValueError(f"param_grid contains invalid hyperparameters: {invalid_params}. Valid keys: {valid_params}")


        # 初始化变量
        best_params = None
        best_auc = -np.inf

        # 生成所有超参数组合
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        print(f"Starting {cv_folds}-fold cross-validation: {len(param_combinations)} parameter combinations to evaluate...")

        for params in param_combinations:
            # Apply current hyperparameter combination to this optimizer instance
            current_params = dict(zip(param_names, params))
            self.regularization_strength = current_params.get('regularization_strength', self.regularization_strength)
            self.n_intial = current_params.get('n_intial', self.n_intial)
            self.noise_scale = current_params.get('noise_scale', self.noise_scale)

            fold_aucs = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx,:], X.iloc[val_idx,:]
                y_train, y_val = y[train_idx], y[val_idx]

                # Re-run optimization on the training fold.
                self.optimize(X_train, y_train)

                # Evaluate optimized betas on the validation fold.
                grs_val = self._calculate_grs(np.array(X_val), self.optimized_betas)
                val_auc = roc_auc_score(y_val, grs_val)
                fold_aucs.append(val_auc)


            # Compute mean AUC across folds for this parameter combination.
            mean_auc = np.mean(fold_aucs)
            print(f"Mean validation AUC for {current_params}: {mean_auc:.4f}")

            # Update best seen hyperparameters if current mean AUC is better.
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = current_params
            
        print(f"Best parameters: {json.dumps(best_params, indent=2)}")
        print(f"Best mean validation AUC: {best_auc:.4f}")

        return {
            'best_params': best_params,
            'best_auc': best_auc
        }