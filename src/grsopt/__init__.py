"""
GRSOpt: A module for optimizing genetic risk scoring (GRS) models.
"""

from .config import OptimizerConfig, PipelineConfig, SelectionConfig
from .grsoptimizer import grsoptimizer
from .metrics import calculate_auc
from .grspruner import backward_elimination
from .draw import grs_pic, roc_pic
from .selection import (
    backward_selection_cv,
    forward_selection_cv,
    rank_snps_by_importance,
    select_optimal_count,
    selection_curve_cv,
)
from .pipeline import SelectionResult, run_selection_pipeline

__all__ = [
    "grsoptimizer",
    "calculate_auc",
    "backward_elimination",
    "grs_pic",
    "roc_pic",
    "OptimizerConfig",
    "SelectionConfig",
    "PipelineConfig",
    "rank_snps_by_importance",
    "selection_curve_cv",
    "forward_selection_cv",
    "backward_selection_cv",
    "select_optimal_count",
    "run_selection_pipeline",
    "SelectionResult",
]
 