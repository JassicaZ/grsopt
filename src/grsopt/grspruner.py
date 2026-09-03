import pandas as pd

"""
Feature selection utilities for genetic risk scoring (GRS).
"""

# ``calculate_auc`` now lives in ``grsopt.metrics``; re-export it here so that
# ``from grsopt.grspruner import calculate_auc`` keeps working.
from .metrics import calculate_auc  # noqa: F401


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
