"""Unit tests for the CV-based feature selection module."""

import numpy as np
import pandas as pd
import pytest

import grsopt
from grsopt.config import OptimizerConfig, PipelineConfig, SelectionConfig


def _make_data(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        columns=[f"SNP_{i}" for i in range(p)],
    )
    latent = sum(
        X.iloc[:, i] * ((i + 1) * 0.2) for i in range(p)
    )
    y = (latent > np.median(latent)).astype(int)
    betas = pd.Series(rng.normal(size=p), index=X.columns)
    return X, y, betas


def _fast_optimizer():
    # Keep the beta optimizer cheap so the tests run quickly.
    return OptimizerConfig(max_iter=20, n_initial=1, regularization_strength=0.01)


def test_forward_selection_cv_returns_curve():
    X, y, betas = _make_data()
    sel = SelectionConfig(direction="forward", cv_folds=3, start_n=2, end_n=6, step=2)
    curve = grsopt.forward_selection_cv(X, y, betas, sel, _fast_optimizer())
    assert isinstance(curve, pd.DataFrame)
    assert {"count", "mean_auc", "std_auc"} <= set(curve.columns)
    assert curve["count"].max() == 6
    assert curve["count"].min() == 2


def test_backward_selection_cv_returns_curve():
    X, y, betas = _make_data()
    sel = SelectionConfig(direction="backward", cv_folds=3, start_n=6, end_n=2, step=2)
    curve = grsopt.backward_selection_cv(X, y, betas, sel, _fast_optimizer())
    assert isinstance(curve, pd.DataFrame)
    assert {"count", "mean_auc", "std_auc"} <= set(curve.columns)
    assert curve["count"].max() == 6
    assert curve["count"].min() == 2


def test_select_optimal_count_se_rule():
    curve = pd.DataFrame(
        {
            "count": [2, 4, 6],
            "mean_auc": [0.6, 0.9, 0.85],
            "std_auc": [0.05, 0.05, 0.05],
        }
    )
    # Best AUC is 0.9 (count 4); 1-SE threshold 0.85 -> smallest count among
    # counts whose mean >= 0.85 is 4.
    assert grsopt.select_optimal_count(curve, choose="auto", se_rule=True) == 4


def test_select_optimal_count_manual():
    curve = pd.DataFrame(
        {"count": [2, 4, 6], "mean_auc": [0.6, 0.9, 0.85], "std_auc": [0.05, 0.05, 0.05]}
    )
    assert grsopt.select_optimal_count(curve, choose="manual", chosen_n=6) == 6
    with pytest.raises(ValueError):
        grsopt.select_optimal_count(curve, choose="manual")


def test_run_selection_pipeline():
    X, y, betas = _make_data()
    cfg = PipelineConfig(
        selection=SelectionConfig(
            direction="forward", cv_folds=3, start_n=2, end_n=4, step=1
        ),
        optimizer=_fast_optimizer(),
    )
    result = grsopt.run_selection_pipeline(X, y, betas, cfg)
    assert isinstance(result, grsopt.SelectionResult)
    assert result.chosen_n is not None
    assert len(result.selected_snps) == result.chosen_n


def test_run_selection_pipeline_plot_only():
    X, y, betas = _make_data()
    cfg = PipelineConfig(
        selection=SelectionConfig(
            direction="forward", cv_folds=3, start_n=2, end_n=4, step=1
        ),
        optimizer=_fast_optimizer(),
        plot_only=True,
    )
    result = grsopt.run_selection_pipeline(X, y, betas, cfg)
    assert result.selected_snps is None
    assert result.chosen_n is None
    assert isinstance(result.curve, pd.DataFrame)


def test_selection_validates_inputs():
    X, y, betas = _make_data()
    with pytest.raises(ValueError):
        grsopt.forward_selection_cv(
            X, y[:10], betas, SelectionConfig(cv_folds=2), _fast_optimizer()
        )
    with pytest.raises(ValueError):
        grsopt.selection_curve_cv(
            X,
            y,
            betas,
            direction="sideways",
            selection_cfg=SelectionConfig(cv_folds=2),
            optimizer_cfg=_fast_optimizer(),
        )
