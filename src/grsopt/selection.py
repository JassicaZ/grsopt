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
        n_intial=optimizer_cfg.n_initial,
        noise_scale=optimizer_cfg.noise_scale,
        loss=optimizer_cfg.loss,
    )
    result = opt.optimize(X_tr[selected], y_tr)
    beta_series = to_beta_series(result["optimized_betas"], selected)
    auc_optimized = calculate_auc(X_va, y_va, beta_series, selected)
    return auc_optimized, auc_original


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
    *within the current subset* and removes the least important one. The
    ranking is recomputed inside the current subset.
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

    Returns a DataFrame with columns ``count``, ``mean_auc``, ``std_auc``,
    ``mean_improvement`` and ``std_improvement``. ``mean_improvement`` is the
    mean across folds of ``optimized AUC - original AUC`` on the validation set.
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

    fold_opt: dict = {}
    fold_orig: dict = {}
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        if direction == "forward":
            subsets = _forward_subsets(X_tr, y_tr, betas, selection_cfg)
        else:
            subsets = _backward_subsets(X_tr, y_tr, betas, selection_cfg)
        for count, selected in subsets:
            auc_opt, auc_orig = fit_optimize_evaluate(
                X_tr, y_tr, X_va, y_va, selected, betas, optimizer_cfg
            )
            key = int(count)
            fold_opt.setdefault(key, []).append(auc_opt)
            fold_orig.setdefault(key, []).append(auc_orig)
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    records = []
    for count in sorted(fold_opt):
        opt_vals = fold_opt[count]
        orig_vals = fold_orig[count]
        improvements = [o - p for o, p in zip(opt_vals, orig_vals)]
        records.append(
            {
                "count": int(count),
                "mean_auc": float(np.mean(opt_vals)),
                "std_auc": float(np.std(opt_vals)),
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
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


def _greedy_backward_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    betas,
    selection_cfg: SelectionConfig,
    optimizer_cfg: OptimizerConfig,
    global_order: Optional[ColumnSubset] = None,
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
                current = candidate
                best_auc = cand_auc
                accepted = True
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
                current = candidate
                best_auc = cand_auc
                accepted = True
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


def greedy_backward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy backward elimination using validation AUC as the accept criterion.

    This performs a wrapper-style greedy backward search: within every fold a
    deletion is accepted only if the optimised validation AUC improves on the
    best subset found so far (by more than ``selection_cfg.margin``). Because the
    validation set is used to accept/reject deletions, the reported validation
    AUC is optimistic; use an external / nested validation for honest evaluation.

    Returns a dict with ``per_fold`` (list of fold results), ``frequency``
    (SNP -> selection frequency), ``selected_snps`` (kept SNPs, by frequency),
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
    logger.info("Running greedy backward selection over %d folds.", len(folds))

    per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        subset, auc = _greedy_backward_fold(
            X_tr, y_tr, X_va, y_va, betas, selection_cfg, optimizer_cfg
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    return _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)


def shared_greedy_backward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy backward selection with a shared ranking across all folds.

    Same accept/reject rule as :func:`greedy_backward_selection_cv` (a deletion is
    kept only if the optimised validation AUC improves on the best subset found so
    far by more than ``selection_cfg.margin``), but the importance ranking is
    computed **once** on the full data and shared by every fold — the deletion
    order is not recomputed inside each fold. Within each fold, deletions are still
    accepted based on that fold's own validation AUC.

    Because the ranking is computed on the full data (which includes each fold's
    validation part), the validation AUCs here are mildly optimistic; report
    performance on an independent external set for honest evaluation.

    Returns the same structure as :func:`greedy_backward_selection_cv`
    (``per_fold``, ``frequency``, ``selected_snps``, ``mean_auc``, ``std_auc``).
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
            global_order=global_order,
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    return _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)


def greedy_forward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy forward selection using validation AUC as the accept criterion.

    Same accept/reject rule as the backward greedy, but SNPs are **added** in
    importance order: a new SNP is kept only if the optimised validation AUC
    improves on the best subset found so far (by more than
    ``selection_cfg.margin``). The ranking is recomputed inside each fold.

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
    logger.info("Running greedy forward selection over %d folds.", len(folds))

    per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X.iloc[va_idx], y_arr[va_idx]
        subset, auc = _greedy_forward_fold(
            X_tr, y_tr, X_va, y_va, betas, selection_cfg, optimizer_cfg
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    return _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)


def shared_greedy_forward_selection_cv(
    X,
    y,
    betas,
    selection_cfg: Optional[SelectionConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
) -> dict:
    """Greedy forward selection with a shared ranking across all folds.

    Same accept/reject rule as :func:`greedy_forward_selection_cv`, but the
    importance ranking is computed **once** on the full data and shared by every
    fold — SNPs are added in that shared order and kept only if the optimised
    validation AUC improves on the best subset found so far (by more than
    ``selection_cfg.margin``).

    Because the ranking is computed on the full data (which includes each fold's
    validation part), the validation AUCs here are mildly optimistic; report
    performance on an independent external set for honest evaluation.

    Returns the same structure as :func:`greedy_forward_selection_cv`.
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
            global_order=global_order,
        )
        per_fold.append({"fold": fold, "selected_snps": subset, "val_auc": auc})
        logger.info("Finished fold %d/%d.", fold + 1, len(folds))

    return _aggregate_frequency(per_fold, X.columns, selection_cfg.retention_threshold)


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
