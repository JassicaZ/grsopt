"""Feature selection for genetic risk score (GRS) models.

This module implements CV-based forward and backward feature selection. It
reuses :func:`grsopt.metrics.calculate_auc` for scoring and
:class:`grsopt.grsoptimizer.grsoptimizer` for beta optimisation. The ranking is
recomputed inside every CV fold so that validation AUC estimates are not
optimistic.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import OptimizerConfig, SelectionConfig
from .grsoptimizer import grsoptimizer
from .metrics import calculate_auc

logger = logging.getLogger(__name__)

ColumnSubset = List[str]


def _coerce_betas(betas, columns) -> pd.Series:
    """Normalise ``betas`` to a Series indexed by ``columns`` for selection."""
    if betas is None:
        raise ValueError("betas must be provided for feature selection.")
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


def _validate_inputs(X, y, betas) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X ({X.shape[0]}) does not match length of y ({len(y)})."
        )
    if betas is None:
        raise ValueError("betas must be provided for feature selection.")
    missing = [c for c in X.columns if c not in betas.index]
    if missing:
        raise ValueError(f"betas is missing coefficients for SNPs {missing}.")
    if np.unique(np.asarray(y)).size < 2:
        raise ValueError("y must contain both classes (0 and 1).")


def to_beta_series(betas_array, columns) -> pd.Series:
    """Convert a raw beta array back to a Series indexed by ``columns``."""
    if len(betas_array) != len(columns):
        raise ValueError(
            f"Length of betas_array ({len(betas_array)}) must match columns ({len(columns)})."
        )
    return pd.Series(betas_array, index=columns)


def _importance_scores(X, y, betas, subset) -> dict:
    """Per-SNP importance within ``subset`` via the rank-one update trick.

    Importance is ``AUC(subset) - AUC(subset minus SNP)``. Computing each
    ``AUC(subset minus SNP)`` via ``GRS_full - X_j * beta_j`` avoids rebuilding
    the full dot product for every SNP.
    """
    subset = list(subset)
    Xs = X[subset].to_numpy()
    beta = betas.loc[subset].to_numpy()
    y_arr = np.asarray(y)
    grs_full = Xs.dot(beta)
    auc_full = roc_auc_score(y_arr, grs_full)
    scores = {}
    for j, snp in enumerate(subset):
        grs_without = grs_full - Xs[:, j] * beta[j]
        auc_without = roc_auc_score(y_arr, grs_without)
        scores[snp] = float(auc_full - auc_without)
    return scores


def rank_snps_by_importance(X, y, betas) -> ColumnSubset:
    """Rank SNPs by their marginal contribution to AUC.

    Higher importance means deleting the SNP causes a larger AUC drop.
    Returns SNP names sorted from most to least important.
    """
    betas = _coerce_betas(betas, X.columns)
    _validate_inputs(X, y, betas)
    scores = _importance_scores(X, y, betas, X.columns)
    return sorted(scores, key=scores.get, reverse=True)


def _within_subset_ranking(X, y, betas, subset) -> ColumnSubset:
    """Rank the SNPs within ``subset`` by importance (used by backward)."""
    scores = _importance_scores(X, y, betas, subset)
    return sorted(subset, key=scores.get, reverse=True)


def fit_optimize_evaluate(
    X_tr, y_tr, X_va, y_va, selected, betas, optimizer_cfg: OptimizerConfig
) -> float:
    """Fit betas on the training fold and evaluate AUC on the validation fold."""
    if not selected:
        raise ValueError("selected feature list must not be empty.")
    missing = [c for c in selected if c not in X_tr.columns]
    if missing:
        raise ValueError(f"Selected SNPs not in training X: {missing}")

    betas = _coerce_betas(betas, X_tr.columns)
    orig = betas.loc[selected].to_numpy()
    opt = grsoptimizer(
        original_betas=orig,
        regularization_strength=optimizer_cfg.regularization_strength,
        max_iter=optimizer_cfg.max_iter,
        n_intial=optimizer_cfg.n_initial,
        noise_scale=optimizer_cfg.noise_scale,
    )
    result = opt.optimize(X_tr[selected], y_tr)
    beta_series = to_beta_series(result["optimized_betas"], selected)
    return calculate_auc(X_va, y_va, beta_series, selected)


def _forward_subsets(
    X, y, betas, config: SelectionConfig
) -> Iterator[Tuple[int, ColumnSubset]]:
    """Yield ``(count, selected)`` for forward selection.

    A single importance ranking is computed on the training data, then SNPs are
    added in descending importance order.
    """
    cols = list(X.columns)
    start_n = config.start_n if config.start_n is not None else 10
    end_n = config.end_n if config.end_n is not None else len(cols)
    start_n = max(1, min(start_n, len(cols)))
    end_n = max(start_n, min(end_n, len(cols)))
    order = rank_snps_by_importance(X, y, betas)
    for c in range(start_n, end_n + 1, config.step):
        yield c, order[:c]


def _backward_subsets(
    X, y, betas, config: SelectionConfig
) -> Iterator[Tuple[int, ColumnSubset]]:
    """Yield ``(count, selected)`` for backward (greedy) selection.

    Starts from the full subset and, at each step, re-ranks the remaining SNPs
    *within the current subset* and removes the least important one. Because the
    ranking is recomputed inside the current subset, this differs from forward
    selection.
    """
    cols = list(X.columns)
    start_n = config.start_n if config.start_n is not None else len(cols)
    if not (0 < start_n <= len(cols)):
        raise ValueError(f"start_n must be in [1, {len(cols)}], got {start_n}.")
    end_n = config.end_n if config.end_n is not None else 10
    end_n = max(1, min(end_n, start_n))

    if config.start_n is None:
        current = list(cols)
    else:
        current = rank_snps_by_importance(X, y, betas)[:start_n]

    removed = 0
    while len(current) >= end_n:
        if removed % config.step == 0:
            yield len(current), list(current)
        if len(current) == end_n:
            break
        worst = _within_subset_ranking(X, y, betas, current)[-1]
        current.remove(worst)
        removed += 1


def selection_curve_cv(
    X,
    y,
    betas,
    direction: str,
    selection_cfg: SelectionConfig,
    optimizer_cfg: OptimizerConfig,
) -> pd.DataFrame:
    """Run CV-based forward/backward selection and return an AUC curve.

    Returns a DataFrame with columns ``count``, ``mean_auc``, ``std_auc``.
    """
    betas = _coerce_betas(betas, X.columns)
    _validate_inputs(X, y, betas)
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"direction must be 'forward' or 'backward', got {direction!r}."
        )
    if selection_cfg.cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)
    skf = StratifiedKFold(
        n_splits=selection_cfg.cv_folds,
        shuffle=True,
        random_state=selection_cfg.random_state,
    )
    folds = list(skf.split(X, y_arr))
    logger.info("Running %s selection over %d folds.", direction, len(folds))

    fold_aucs: dict = {}
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        if direction == "forward":
            subsets = _forward_subsets(X_tr, y_tr, betas, selection_cfg)
        else:
            subsets = _backward_subsets(X_tr, y_tr, betas, selection_cfg)
        for count, selected in subsets:
            auc = fit_optimize_evaluate(
                X_tr, y_tr, X_va, y_va, selected, betas, optimizer_cfg
            )
            fold_aucs.setdefault(int(count), []).append(auc)
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    records = []
    for count in sorted(fold_aucs):
        vals = fold_aucs[count]
        records.append(
            {
                "count": int(count),
                "mean_auc": float(np.mean(vals)),
                "std_auc": float(np.std(vals)),
            }
        )
    return pd.DataFrame(records)


def forward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> pd.DataFrame:
    """Convenience wrapper for forward selection CV."""
    selection_cfg = selection_cfg or SelectionConfig()
    optimizer_cfg = optimizer_cfg or OptimizerConfig()
    return selection_curve_cv(X, y, betas, "forward", selection_cfg, optimizer_cfg)


def backward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> pd.DataFrame:
    """Convenience wrapper for backward selection CV."""
    selection_cfg = selection_cfg or SelectionConfig(direction="backward")
    optimizer_cfg = optimizer_cfg or OptimizerConfig()
    return selection_curve_cv(X, y, betas, "backward", selection_cfg, optimizer_cfg)


def select_optimal_count(
    curve: pd.DataFrame,
    choose: str = "auto",
    chosen_n: Optional[int] = None,
    se_rule: bool = True,
) -> int:
    """Choose the number of SNPs from a selection curve.

    Parameters
    ----------
    curve : pd.DataFrame
        Must contain columns ``count``, ``mean_auc`` and ``std_auc``.
    choose : {'auto', 'manual'}, default 'auto'
        ``auto`` selects automatically (1-SE rule if ``se_rule`` otherwise the
        max), ``manual`` uses ``chosen_n``.
    chosen_n : int, optional
        Required when ``choose='manual'``.
    se_rule : bool, default True
        When ``choose='auto'``, use the 1-standard-error rule to pick the most
        parsimonious count within one SE of the best AUC.

    Returns
    -------
    int
        The chosen number of SNPs.
    """
    required = {"count", "mean_auc", "std_auc"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"curve is missing required columns: {missing}")

    counts = curve["count"].to_numpy()
    means = curve["mean_auc"].to_numpy()
    stds = curve["std_auc"].to_numpy()

    if choose == "manual":
        if chosen_n is None:
            raise ValueError("choose='manual' requires chosen_n.")
        if int(chosen_n) not in counts:
            raise ValueError(
                f"chosen_n={chosen_n} is not among evaluated counts {counts.tolist()}."
            )
        return int(chosen_n)
    if choose != "auto":
        raise ValueError(f"choose must be 'auto' or 'manual', got {choose!r}.")

    best_i = int(np.argmax(means))
    if se_rule:
        threshold = means[best_i] - stds[best_i]
        candidates = np.where(means >= threshold)[0]
        chosen_i = int(candidates[0])
    else:
        chosen_i = best_i
    return int(counts[chosen_i])
