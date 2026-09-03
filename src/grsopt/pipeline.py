"""End-to-end feature selection pipeline for GRS models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import PipelineConfig
from .selection import (
    rank_snps_by_importance,
    select_optimal_count,
    selection_curve_cv,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of :func:`run_selection_pipeline`."""

    selected_snps: Optional[list]
    chosen_n: Optional[int]
    curve: pd.DataFrame
    ordering: Optional[list]
    direction: str


def run_selection_pipeline(X, y, betas, config: PipelineConfig) -> SelectionResult:
    """Run CV-based feature selection and return the chosen SNP set.

    When ``config.plot_only`` is true, only the CV curve is produced and
    ``selected_snps``/``chosen_n`` are ``None``, letting the user inspect the
    curve and re-run with ``choose='manual'`` and an explicit ``chosen_n``.
    """
    sel_cfg = config.selection
    curve = selection_curve_cv(
        X, y, betas, sel_cfg.direction, sel_cfg, config.optimizer
    )

    if config.plot_only:
        return SelectionResult(
            selected_snps=None,
            chosen_n=None,
            curve=curve,
            ordering=None,
            direction=sel_cfg.direction,
        )

    chosen_n = select_optimal_count(
        curve,
        choose=config.choose,
        chosen_n=config.chosen_n,
        se_rule=sel_cfg.se_rule,
    )
    ordering = rank_snps_by_importance(X, y, betas)
    selected_snps = ordering[:chosen_n]

    logger.info(
        "Chose %d SNPs (%s, direction=%s).",
        chosen_n,
        config.choose,
        sel_cfg.direction,
    )

    if config.plot:
        from .draw import plot_selection_curve

        plot_selection_curve(curve, chosen_n=chosen_n)

    return SelectionResult(
        selected_snps=selected_snps,
        chosen_n=chosen_n,
        curve=curve,
        ordering=ordering,
        direction=sel_cfg.direction,
    )
