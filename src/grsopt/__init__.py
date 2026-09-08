"""
GRSOpt: A module for optimizing genetic risk scoring (GRS) models.
"""

from .config import OptimizerConfig, SelectionConfig
from .grsoptimizer import grsoptimizer
from .metrics import calculate_auc
from .grspruner import backward_elimination
from .draw import grs_pic, roc_pic
from .selection import backward_selection, forward_selection, rank_snps_by_importance

__all__ = [
    "grsoptimizer",
    "calculate_auc",
    "backward_elimination",
    "grs_pic",
    "roc_pic",
    "OptimizerConfig",
    "SelectionConfig",
    "rank_snps_by_importance",
    "forward_selection",
    "backward_selection",
]
 