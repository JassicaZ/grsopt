"""Unit tests for the greedy CV-based feature selection functions."""

import numpy as np
import pandas as pd
import pytest

import grsopt
from grsopt.config import OptimizerConfig, SelectionConfig


def _make_data(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        columns=[f"SNP_{i}" for i in range(p)],
    )
    latent = sum(X.iloc[:, i] * ((i + 1) * 0.2) for i in range(p))
    y = (latent > np.median(latent)).astype(int)
    betas = pd.Series(rng.normal(size=p), index=X.columns)
    return X, y, betas


def _fast_optimizer():
    # Keep the beta optimizer cheap so the tests run quickly.
    return OptimizerConfig(max_iter=20, n_initial=1, regularization_strength=0.01)


def test_forward_selection():
    X, y, betas = _make_data()
    sel = SelectionConfig(
        direction="forward", cv_folds=3, start_n=2, end_n=6, margin=0.0,
        retention_threshold=0.5,
    )
    res = grsopt.forward_selection(X, y, betas, sel, _fast_optimizer())
    assert isinstance(res, dict)
    assert "per_fold" in res and len(res["per_fold"]) == 3
    assert "frequency" in res and "selected_snps" in res
    assert "mean_auc" in res and "std_auc" in res
    for rec in res["per_fold"]:
        assert 2 <= len(rec["selected_snps"]) <= 6


def test_backward_selection():
    X, y, betas = _make_data()
    sel = SelectionConfig(
        direction="backward", cv_folds=3, start_n=6, end_n=2, margin=0.0,
        retention_threshold=0.5,
    )
    res = grsopt.backward_selection(X, y, betas, sel, _fast_optimizer())
    assert isinstance(res, dict)
    assert "per_fold" in res and len(res["per_fold"]) == 3
    assert "frequency" in res and "selected_snps" in res
    assert "mean_auc" in res and "std_auc" in res
    for rec in res["per_fold"]:
        assert 2 <= len(rec["selected_snps"]) <= 6


def test_selection_validates_inputs():
    X, y, betas = _make_data()
    with pytest.raises(ValueError):
        grsopt.forward_selection(
            X, y[:10], betas, SelectionConfig(cv_folds=2), _fast_optimizer()
        )
