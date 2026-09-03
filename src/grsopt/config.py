"""Central configuration for grsopt experiments.

Configuration is grouped into small dataclasses so different experiments can be
reproduced by constructing a config object and passing it through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerConfig:
    """Configuration for the GRS beta optimizer (``grsoptimizer``)."""

    regularization_strength: float = 0.01
    max_iter: int = 2000
    n_initial: int = 5
    noise_scale: float = 0.3
    random_state: Optional[int] = None


@dataclass
class SelectionConfig:
    """Configuration for the feature-selection CV loop.

    ``start_n`` / ``end_n`` are resolved direction-aware by the selection
    routines:

    * forward: ``start_n`` defaults to 10, ``end_n`` defaults to all SNPs.
    * backward: ``start_n`` defaults to all SNPs, ``end_n`` defaults to 10.
    """

    direction: str = "forward"
    cv_folds: int = 5
    random_state: int = 42
    start_n: Optional[int] = None
    end_n: Optional[int] = None
    step: int = 1
    se_rule: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for :func:`run_selection_pipeline`."""

    selection: SelectionConfig = field(default_factory=SelectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    choose: str = "auto"             # 'auto' or 'manual'
    chosen_n: Optional[int] = None   # required when choose='manual'
    plot: bool = False               # plot the curve at the end
    plot_only: bool = False          # return only the curve (no selection)
