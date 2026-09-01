import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

"""
For feature selection in genetic risk scoring (GRS).
"""


def calculate_auc(X, y,selected_features='all'):
    """
    Calculate the auc of selected features from backward elimination.
    
    Parameters
    ----------
   X : DataFrame
        Geno dataframe. SNPs in columns and individuals in rows.
        The values are snp * beta.
    y : list
        The state of individual (disease or healthy).
        With the same order as the rows in X.
    selected_features : str or list
        A list of SNPs to evaluate. 'all': use all SNPs in X columns.
    
    Returns
    -------
    float
        The mean auc of the optimized model.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.shape[0] != len(y):
        raise ValueError(f"Number of rows in X ({X.shape[0]}) does not match length of y ({len(y)}).")

    if selected_features=='all':
        selected_features= X.columns.tolist()
    elif any(f not in X.columns for f in selected_features):
        missing = [f for f in selected_features if f not in X.columns]
        raise ValueError(f"Find feactures not exist in X: {missing}")

    grs = X[selected_features].sum(axis=1)
    return roc_auc_score(np.array(y), grs)



def backward_elimination(X, y, min_features=5):
    """
    Perform backward elimination to select optimal SNP subset.

    Parameters
    ----------
    X : DataFrame
        Geno dataframe. SNPs in columns and individuals in rows.
        The values are snp * beta.
    y : list
        The state of individual (disease or healthy).
        With the same order as the rows in X.
    min_features : int (default=5)
        Minimum number of features to retain.

    Returns
    -------
    selected : list
        List of selected SNPs.
    history : list
        The eliminated SNP and AUC at each iteration.
    """
    selected = list(X.columns)
    best_auc = calculate_auc(X, y, selected)
    history = [(selected.copy(), best_auc)]

    print(f"Initial AUC with all features ({len(selected)} loci): {best_auc:.4f}")

    while len(selected) > min_features:
        auc_drop = []
        for feat in selected:
            trial = [f for f in selected if f != feat]
            auc = calculate_auc(X, y, trial)
            auc_drop.append((feat, auc))

        # Find the feature whose removal yields the highest AUC (i.e. contributes nothing or negatively)
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
    # Simulate data
    np.random.seed(42)
    n_samples = 100
    n_snps = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_snps),
        columns=[f'SNP_{i}' for i in range(n_snps)]
    )  # Simulated values already multiplied by beta

    y = np.random.choice([0, 1], size=n_samples)  # 0: healthy, 1: disease

    # Test calculate_auc
    selected_features = X.columns[:5].tolist()
    auc = calculate_auc(X, y, selected_features)
    print(f'Test AUC for selected features: {auc:.4f}')

    # Test backward_elimination
    selected, history = backward_elimination(X, y, min_features=3)
    print(f'Backward elimination selected features: {selected}')
    print('Elimination history:')
    for feats, auc_val in history:
        print(f'Features: {feats}, AUC: {auc_val:.4f}')