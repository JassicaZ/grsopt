"""Central configuration for grsopt experiments.

Configuration is grouped into small dataclasses so different experiments can be
reproduced by constructing a config object and passing it through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizerConfig:
    """Configuration for the GRS beta optimizer (``grsoptimizer``).

    ``loss`` selects the objective: ``'log'`` for the logistic cross-entropy or
    ``'rank'`` for the pairwise logistic (AUC) rank loss.
    """

    regularization_strength: float = 0.01
    max_iter: int = 2000
    n_initial: int = 5
    noise_scale: float = 0.3
    loss: str = "rank"


@dataclass
class SelectionConfig:
    """Configuration for the greedy feature-selection CV loop.

    ``start_n`` / ``end_n`` are resolved direction-aware:

    * forward: ``start_n`` defaults to 10, ``end_n`` defaults to all SNPs.
    * backward: ``start_n`` defaults to all SNPs, ``end_n`` defaults to 10.
    """

    direction: str = "forward"
    cv_folds: int = 5
    random_state: int = 42
    start_n: Optional[int] = None
    end_n: Optional[int] = None
    margin: float = 0.0                 # min validation-AUC improvement to accept a change
    retention_threshold: float = 0.5    # min selection frequency to keep a SNP
