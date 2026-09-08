"""Feature selection for genetic risk score (GRS) models.

This module implements CV-based forward and backward feature selection.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import OptimizerConfig, SelectionConfig
from .grsoptimizer import grsoptimizer
from .metrics import as_beta_series, calculate_auc

logger = logging.getLogger(__name__)

ColumnSubset = List[str]


def _coerce_betas(betas, columns) -> pd.Series:
    """Normalise ``betas`` to a Series indexed by ``columns`` for selection."""
    if betas is None:
        raise ValueError("betas must be provided for feature selection.")
    return as_beta_series(betas, columns)


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
) -> Tuple[float, float]:
    """Fit betas on the training fold and evaluate AUC on the validation fold.

    Returns ``(optimized_auc, original_auc)``. ``optimized_auc`` is the
    validation AUC of the model fitted on the training fold; ``original_auc`` is
    the validation AUC computed with the original (un-optimised) betas.
    """
    if not selected:
        raise ValueError("selected feature list must not be empty.")
    missing = [c for c in selected if c not in X_tr.columns]
    if missing:
        raise ValueError(f"Selected SNPs not in training X: {missing}")

    betas = _coerce_betas(betas, X_tr.columns)

    # Validation AUC of the original (un-optimised) betas.
    original_beta_series = betas.loc[selected]
    auc_original = calculate_auc(X_va, y_va, original_beta_series, selected)

    orig = betas.loc[selected].to_numpy()
    opt = grsoptimizer(
        original_betas=orig,
        regularization_strength=optimizer_cfg.regularization_strength,
        max_iter=optimizer_cfg.max_iter,
        n_initial=optimizer_cfg.n_initial,
        noise_scale=optimizer_cfg.noise_scale,
        loss=optimizer_cfg.loss,
        verbose=optimizer_cfg.verbose,
    )
    result = opt.optimize(X_tr[selected], y_tr)
    beta_series = to_beta_series(result["optimized_betas"], selected)
    auc_optimized = calculate_auc(X_va, y_va, beta_series, selected)
    return auc_optimized, auc_original


def _greedy_backward_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    betas,
    selection_cfg: SelectionConfig,
    optimizer_cfg: OptimizerConfig,
    global_order: Optional[ColumnSubset] = None,
    fold: int = 0,
) -> Tuple[ColumnSubset, float]:
    """Greedy backward elimination within a single fold.

    Deletions are tried from the least important SNP upward and accepted only if
    the optimised validation AUC improves on the best subset found so far by more
    than ``selection_cfg.margin``. If ``global_order`` is provided, a shared
    ranking drives the deletion order; otherwise the ranking is recomputed inside
    the current subset of this fold. Returns ``(subset, val_auc)``.
    """
    cols = list(X_tr.columns)
    start_n = selection_cfg.start_n if selection_cfg.start_n is not None else len(cols)
    if not (0 < start_n <= len(cols)):
        raise ValueError(f"start_n must be in [1, {len(cols)}], got {start_n}.")
    end_n = selection_cfg.end_n if selection_cfg.end_n is not None else 10
    end_n = max(1, min(end_n, start_n))

    if selection_cfg.start_n is None:
        current = list(cols)
    else:
        if global_order is not None:
            current = global_order[:start_n]
        else:
            current = rank_snps_by_importance(X_tr, y_tr, betas)[:start_n]

    # Best model found so far: the full/start subset optimised on the training fold.
    best_auc = fit_optimize_evaluate(
        X_tr, y_tr, X_va, y_va, current, betas, optimizer_cfg
    )[0]

    while len(current) > end_n:
        if global_order is not None:
            # Try deletions from the least important (lowest global rank) upward.
            position = {s: i for i, s in enumerate(global_order)}
            try_order = sorted(current, key=lambda s: position[s], reverse=True)
        else:
            # Order from most to least important; try deletions least-first.
            try_order = list(reversed(_within_subset_ranking(X_tr, y_tr, betas, current)))
        accepted = False
        for snp in try_order:
            candidate = [s for s in current if s != snp]
            cand_auc = fit_optimize_evaluate(
                X_tr, y_tr, X_va, y_va, candidate, betas, optimizer_cfg
            )[0]
            if cand_auc > best_auc + selection_cfg.margin:
                old_auc = best_auc
                current = candidate
                best_auc = cand_auc
                accepted = True
                print(f"[fold {fold}] removed {snp}: validation AUC {old_auc:.4f} -> {cand_auc:.4f} (Δ {cand_auc - old_auc:+.4f})")
                break
        if not accepted:
            break

    return current, best_auc


def _greedy_forward_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    betas,
    selection_cfg: SelectionConfig,
    optimizer_cfg: OptimizerConfig,
    global_order: Optional[ColumnSubset] = None,
    fold: int = 0,
) -> Tuple[ColumnSubset, float]:
    """Greedy forward selection within a single fold.

    SNPs are added in importance order (most important first) and accepted only
    if the optimised validation AUC improves on the best subset found so far by
    more than ``selection_cfg.margin``. If ``global_order`` is provided, a shared
    ranking drives the addition order; otherwise the ranking is recomputed inside
    this fold. Returns ``(subset, val_auc)``.
    """
    cols = list(X_tr.columns)
    start_n = selection_cfg.start_n if selection_cfg.start_n is not None else 10
    end_n = selection_cfg.end_n if selection_cfg.end_n is not None else len(cols)
    start_n = max(1, min(start_n, len(cols)))
    end_n = max(start_n, min(end_n, len(cols)))

    if global_order is not None:
        order = global_order
    else:
        order = rank_snps_by_importance(X_tr, y_tr, betas)

    current = order[:start_n]

    # Best model found so far: the starting subset optimised on the training fold.
    best_auc = fit_optimize_evaluate(
        X_tr, y_tr, X_va, y_va, current, betas, optimizer_cfg
    )[0]

    while len(current) < end_n:
        # Candidates are the SNPs not yet added, most important first.
        candidates = [s for s in order if s not in current]
        accepted = False
        for snp in candidates:
            candidate = current + [snp]
            cand_auc = fit_optimize_evaluate(
                X_tr, y_tr, X_va, y_va, candidate, betas, optimizer_cfg
            )[0]
            if cand_auc > best_auc + selection_cfg.margin:
                old_auc = best_auc
                current = candidate
                best_auc = cand_auc
                accepted = True
                print(f"[fold {fold}] added {snp}: validation AUC {old_auc:.4f} -> {cand_auc:.4f} (Δ {cand_auc - old_auc:+.4f})")
                break
        if not accepted:
            break

    return current, best_auc


def _aggregate_frequency(
    per_fold, columns, retention_threshold: float
) -> dict:
    """Aggregate per-fold greedy results into frequencies and summary stats."""
    n_folds = len(per_fold)
    counts = {c: 0 for c in columns}
    for rec in per_fold:
        for snp in rec["selected_snps"]:
            counts[snp] += 1
    frequency = {s: c / n_folds for s, c in counts.items()}
    selected_snps = sorted(
        (s for s, f in frequency.items() if f >= retention_threshold),
        key=lambda s: frequency[s],
        reverse=True,
    )
    aucs = [rec["val_auc"] for rec in per_fold]
    return {
        "per_fold": per_fold,
        "frequency": frequency,
        "selected_snps": selected_snps,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }


def backward_selection(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy backward selection with a shared ranking across all folds.

    A deletion is kept only if the optimised validation AUC improves on the best
    subset found so far by more than ``selection_cfg.margin``; the importance
    ranking is computed **once** on the full data and shared by every fold. The
    ranking is not recomputed inside each fold, but each fold's deletions are still
    accepted based on its own validation AUC.

    Because the ranking is computed on the full data (which includes each fold's
    validation part), the validation AUCs here are mildly optimistic; report
    performance on an independent external set for honest evaluation.

    Returns a dict with ``per_fold``, ``frequency``, ``selected_snps``,
    ``mean_auc`` and ``std_auc``.
    """
    selection_cfg = selection_cfg or SelectionConfig(direction="backward")
    optimizer_cfg = optimizer_cfg or OptimizerConfig()

    betas = _coerce_betas(betas, X.columns)
    _validate_inputs(X, y, betas)
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

    # Shared ranking computed once on the full data.
    global_order = rank_snps_by_importance(X, y_arr, betas)
    logger.info("Running shared greedy backward selection over %d folds.", len(folds))

    per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        subset, auc = _greedy_backward_fold(
            X_tr, y_tr, X_va, y_va, betas, selection_cfg, optimizer_cfg,
            global_order=global_order, fold=fold,
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    result = _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)
    print(f"Final selected SNPs count: {len(result['selected_snps'])}")
    return result


def forward_selection(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy forward selection with a shared ranking across all folds.

    SNPs are added in the shared importance order and kept only if the optimised
    validation AUC improves on the best subset found so far by more than
    ``selection_cfg.margin``; the ranking is computed **once** on the full data and
    shared by every fold.

    Because the ranking is computed on the full data (which includes each fold's
    validation part), the validation AUCs here are mildly optimistic; report
    performance on an independent external set for honest evaluation.

    Returns a dict with ``per_fold``, ``frequency``, ``selected_snps``,
    ``mean_auc`` and ``std_auc``.
    """
    selection_cfg = selection_cfg or SelectionConfig(direction="forward")
    optimizer_cfg = optimizer_cfg or OptimizerConfig()

    betas = _coerce_betas(betas, X.columns)
    _validate_inputs(X, y, betas)
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

    # Shared ranking computed once on the full data.
    global_order = rank_snps_by_importance(X, y_arr, betas)
    logger.info("Running shared greedy forward selection over %d folds.", len(folds))

    per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        subset, auc = _greedy_forward_fold(
            X_tr, y_tr, X_va, y_va, betas, selection_cfg, optimizer_cfg,
            global_order=global_order, fold=fold,
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    result = _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)
    print(f"Final selected SNPs count: {len(result['selected_snps'])}")
    return result
