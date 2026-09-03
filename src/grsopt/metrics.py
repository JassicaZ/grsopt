"""Scoring metrics for genetic risk score (GRS) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _as_beta_series(betas, columns) -> pd.Series:
    """Normalise ``betas`` to a pandas Series indexed by ``columns``.

    Accepts ``None`` (uniform weights), a pandas Series, a single-column
    DataFrame, or a 1-D array-like whose length matches ``columns``.
    """
    if betas is None:
        return pd.Series(1.0, index=columns)
    if isinstance(betas, pd.DataFrame):
        if betas.shape[1] != 1:
            raise ValueError(
                f"betas DataFrame must have exactly one column, got {betas.shape[1]}."
            )
        betas = betas.iloc[:, 0]
    if isinstance(betas, pd.Series):
        out = betas.reindex(columns)
        if out.isna().any():
            missing = columns[out.isna().to_numpy()].tolist()
            raise ValueError(f"betas is missing coefficients for SNPs {missing}.")
        return out
    if len(betas) != len(columns):
        raise ValueError(
            f"Length of betas ({len(betas)}) must match number of columns ({len(columns)})."
        )
    return pd.Series(np.asarray(betas), index=columns)


def calculate_auc(X, y, betas=None, selected_features="all") -> float:
    """Compute the AUC of a GRS built from ``selected_features``.

    Parameters
    ----------
    X : pd.DataFrame
        Genotype matrix with SNPs in columns and individuals in rows.
    y : array-like
        Binary disease status (0 = healthy, 1 = disease).
    betas : pd.Series, pd.DataFrame, array-like or None, default None
        Beta coefficients indexed by SNP name. If ``None``, an unweighted GRS
        (all betas equal to 1) is used.
    selected_features : str or list of str, default 'all'
        SNPs to evaluate. Use 'all' to include every column in ``X``.

    Returns
    -------
    float
        Area under the ROC curve for the selected feature set.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X ({X.shape[0]}) does not match length of y ({len(y)})."
        )

    if selected_features == "all":
        selected_features = X.columns.tolist()
    elif any(f not in X.columns for f in selected_features):
        missing = [f for f in selected_features if f not in X.columns]
        raise ValueError(f"Features not found in X: {missing}")

    b = _as_beta_series(betas, selected_features)
    grs = np.dot(
        X[selected_features].to_numpy(), b.loc[selected_features].to_numpy()
    )
    return roc_auc_score(np.asarray(y), grs)
