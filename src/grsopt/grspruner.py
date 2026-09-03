import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

"""
Feature selection utilities for genetic risk scoring (GRS).
"""


def calculate_auc(X, y, betas, selected_features='all'):
    """
    Calculate the AUC for a selected subset of SNP features.

    Parameters
    ----------
    X : pd.DataFrame
        Genotype matrix with SNPs in columns and individuals in rows.
    y : array-like
        Binary disease status aligned with the rows in `X`.
        Typically 0 = healthy and 1 = disease.
    betas : pd.DataFrame or pd.Series
        Beta coefficients indexed by SNP name. The index must match the SNP
        names in `X.columns`, using the same order as the selected columns.
        For example, `betas.index` should contain values such as 'SNP_1',
        'SNP_2', ... and match the SNP order in `X`.
    selected_features : str or list of str, default='all'
        SNPs to evaluate. Use 'all' to include every column in `X`.

    Returns
    -------
    float
        Area under the ROC curve (AUC) for the selected feature set.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.shape[0] != len(y):
        raise ValueError(f"Number of rows in X ({X.shape[0]}) does not match length of y ({len(y)}).")

    # Use all SNPs if no subset is specified.
    if selected_features == 'all':
        selected_features = X.columns.tolist()
    elif any(f not in X.columns for f in selected_features):
        missing = [f for f in selected_features if f not in X.columns]
        raise ValueError(f"Features not found in X: {missing}")

    # Compute GRS using the SNP-specific beta values aligned to the selected features.
    grs = np.dot(X[selected_features], betas.loc[selected_features])
    return roc_auc_score(np.array(y), grs)



def backward_elimination(X, y, betas=None, min_features=5):
    """
    Perform backward elimination to select an optimal SNP subset.

    Parameters
    ----------
    X : pd.DataFrame
        Genotype matrix with SNPs in columns and individuals in rows.
    y : array-like
        Binary disease labels aligned with the rows in `X`.
    betas : pd.DataFrame, pd.Series, array-like or None, default None
        SNP-indexed beta values. If ``None``, an unweighted GRS (all betas
        equal to 1) is used.
    min_features : int, default=5
        Minimum number of features to retain during elimination.

    Returns
    -------
    selected : list
        Final selected SNP names.
    history : list
        A record of feature subsets and the corresponding AUC after each
        elimination step.
    """
    if betas is None:
        betas = pd.Series(1.0, index=X.columns)
    elif isinstance(betas, pd.DataFrame):
        betas = betas.iloc[:, 0]

    selected = list(X.columns)
    best_auc = calculate_auc(X, y, betas, selected)
    history = [(selected.copy(), best_auc)]

    print(f"Initial AUC with all features ({len(selected)} loci): {best_auc:.4f}")

    while len(selected) > min_features:
        auc_drop = []
        for feat in selected:
            trial = [f for f in selected if f != feat]
            auc = calculate_auc(X, y, betas, trial)
            auc_drop.append((feat, auc))

        # Remove the SNP whose deletion yields the highest AUC; this indicates the
        # feature contributes little or negatively to discrimination.
        worst_feature, new_auc = max(auc_drop, key=lambda x: x[1])

        if new_auc > best_auc:
            selected.remove(worst_feature)
            best_auc = new_auc
            history.append((selected.copy(), best_auc))
            print(f"Dropped {worst_feature}, AUC improved to {best_auc:.4f}")
        else:
            print("No further AUC improvement by deleting features.")
            break

    return selected, history





if __name__ == '__main__':
    # Simulate a genotype matrix with SNP columns.
    np.random.seed(42)
    n_samples = 100
    n_snps = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_snps),
        columns=[f'SNP_{i}' for i in range(n_snps)]
    )

    # Simulated beta coefficients keyed by SNP name and aligned with X.columns.
    betas = pd.DataFrame(
        {"beta": np.random.randn(n_snps)},
        index=[f'SNP_{i}' for i in range(n_snps)]
    )

    y = np.random.choice([0, 1], size=n_samples)  # 0 = healthy, 1 = disease

    # Test the AUC calculation on a selected subset of SNPs.
    selected_features = X.columns[:5].tolist()
    auc = calculate_auc(X, y, betas, selected_features)
    print(f'Test AUC for selected features: {auc:.4f}')

    # Test backward elimination using the SNP-indexed beta coefficients.
    selected, history = backward_elimination(X, y, betas, min_features=3)
    print(f'Backward elimination selected features: {selected}')
    print('Elimination history:')
    for feats, auc_val in history:
        print(f'Features: {feats}, AUC: {auc_val:.4f}')