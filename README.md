# grsopt

A lightweight Python package for optimizing genetic risk score (GRS) models. `grsopt` supports beta optimization for genetic risk scores, CV-based forward/backward SNP selection, and visualization of GRS distributions and ROC curves.

## Key Features

- Optimize published GRS weights using gradient-based methods
- CV-based **forward** and **backward** SNP feature selection (ranking recomputed inside every fold to avoid optimistic estimates)
- Automatic (1-SE rule) or manual selection of the number of SNPs from the AUC curve
- Centralized experiment configuration via dataclasses for reproducible runs
- Plot GRS score distributions, ROC curves, and SNP-selection curves

## Installation

Install from source:

```bash
git clone https://github.com/JassicaZ/grsopt.git
cd grsopt
pip install .
```

## Quick Start

```python
import grsopt
from grsopt import PipelineConfig, SelectionConfig, OptimizerConfig

# X: genotype DataFrame (individuals x SNPs), y: binary labels (0/1)
# betas: original GRS coefficients indexed by SNP name

cfg = PipelineConfig(
    selection=SelectionConfig(direction="forward", cv_folds=5, start_n=10, step=1),
    optimizer=OptimizerConfig(max_iter=500, n_initial=3),   # reduce cost
)

# 1) Plot the AUC curve first, inspect, then re-run with a chosen count.
result = grsopt.run_selection_pipeline(X, y, betas, cfg)
print(result.selected_snps)   # final selected SNP list
print(result.curve)           # count / mean_auc / std_auc
```

Two-stage workflow (inspect the curve, then choose):

```python
from grsopt import PipelineConfig, SelectionConfig, OptimizerConfig

base = dict(selection=SelectionConfig(direction="backward", cv_folds=5), optimizer=OptimizerConfig(max_iter=500, n_initial=3))
# Stage 1: only plot / return the curve
cfg = PipelineConfig(plot_only=True, **base)
result = grsopt.run_selection_pipeline(X, y, betas, cfg)
# Stage 2: pick the count by eye
cfg = PipelineConfig(choose="manual", chosen_n=20, **base)
result = grsopt.run_selection_pipeline(X, y, betas, cfg)
```

## API

### Configuration (`grsopt.config`)

- `OptimizerConfig`: `regularization_strength`, `max_iter`, `n_initial`, `noise_scale`, `random_state`
- `SelectionConfig`: `direction` (`'forward'`/`'backward'`), `cv_folds`, `random_state`, `start_n`, `end_n`, `step`, `se_rule`
- `PipelineConfig`: `selection`, `optimizer`, `choose` (`'auto'`/`'manual'`), `chosen_n`, `plot`, `plot_only`

### Beta optimization (`grsopt.grsoptimizer`)

Class for optimizing GRS beta weights.

#### Parameters

- `original_betas`: array-like, original published GRS coefficients in the same column order as `X`.
- `regularization_strength`: float, regularization weight to control deviation from original betas
- `max_iter`: int, maximum number of optimization iterations
- `n_intial`: int, number of random initializations to reduce local optimum risk
- `noise_scale`: float, scale of random perturbation for initial beta values

#### Methods

- `optimize(X, y)`: optimize betas on training data and return a result dictionary. `X` is a genotype matrix (rows = samples, columns = SNPs, encoded as `0/1/2`).
- `cross_validate(X, y, param_grid, cv_folds=5, random_state=42)`: evaluate hyperparameter combinations using stratified CV.

### Scoring (`grsopt.metrics`)

- `calculate_auc(X, y, betas=None, selected_features='all')`: AUC of a GRS built from a subset of SNPs. If `betas` is `None`, an unweighted GRS (all betas = 1) is used.

### Feature selection (`grsopt.selection`)

- `rank_snps_by_importance(X, y, betas)`: rank SNPs by their marginal AUC contribution (rank-one update trick).
- `forward_selection_cv(X, y, betas, selection_cfg=None, optimizer_cfg=None)`: forward selection AUC curve.
- `backward_selection_cv(X, y, betas, selection_cfg=None, optimizer_cfg=None)`: backward (greedy, within-subset re-ranking) AUC curve.
- `selection_curve_cv(X, y, betas, direction, selection_cfg, optimizer_cfg)`: shared CV loop.
- `select_optimal_count(curve, choose='auto', chosen_n=None, se_rule=True)`: choose SNP count from the curve (1-SE or manual).

### Pipeline (`grsopt.pipeline`)

- `run_selection_pipeline(X, y, betas, config)`: end-to-end selection, returns a `SelectionResult` (`selected_snps`, `chosen_n`, `curve`, `ordering`, `direction`).

### Plotting (`grsopt.draw`)

- `grs_pic(GRS, model_name, save_path=None)`: plot the GRS distribution by disease label. `GRS` is a DataFrame with columns `grs` and `Disease`.
- `roc_pic(GRS, model_name, save_path=None)`: plot the ROC curve and display AUC.
- `plot_selection_curve(curve, chosen_n=None, ...)`: plot the AUC vs number-of-SNPs curve.

## Testing

Run the included unit tests with:

```bash
pytest
```

## Notes

- 🚧 This repository is **currently under development**. The core functionality is available, while additional features, examples, and documentation are being added and modified.

- 🔒 **No example genotype data** is provided here due to privacy and ethical restrictions.
- 🧪 The package has currently been validated on our internal datasets. If you test it on independent datasets, we would greatly appreciate your feedback.
- 💬 If you encounter any issues, unexpected results, or are interested in collaboration, please open an Issue or contact us.