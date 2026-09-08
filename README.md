# grsopt

A lightweight Python package for optimizing genetic risk score (GRS) models. `grsopt` supports beta optimization for genetic risk scores, CV-based forward/backward SNP selection, and visualization of GRS distributions and ROC curves.

## Key Features

- Optimize published GRS weights using gradient-based methods
- CV-based **forward** or **backward** SNP feature selection 
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
from grsopt import SelectionConfig, OptimizerConfig

# X: genotype DataFrame (individuals x SNPs), y: binary labels (0/1)
# betas: original GRS coefficients indexed by SNP name

cfg = SelectionConfig(
    direction="forward", cv_folds=5, start_n=10, end_n=None,
    margin=0.0, retention_threshold=0.9,
)
opt = OptimizerConfig(max_iter=500, n_initial=3)

result = grsopt.forward_selection(X, y, betas, cfg, opt)
print(result["selected_snps"])   # final selected SNP list
print(result["mean_auc"])        # cross-fold validation AUC (mildly optimistic)
```

## API

### Configuration (`grsopt.config`)

- `OptimizerConfig`: `regularization_strength`, `max_iter`, `n_initial`, `noise_scale`, `loss`, `random_state`
- `SelectionConfig`: `direction` (`'forward'`/`'backward'`), `cv_folds`, `random_state`, `start_n`, `end_n`, `margin`, `retention_threshold`

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

Both return a dict with `per_fold`, `frequency`, `selected_snps`, `mean_auc` and `std_auc`.

- `forward_selection(X, y, betas, selection_cfg=None, optimizer_cfg=None)`: greedy forward selection using a shared ranking; SNPs are added and kept only if the optimised validation AUC improves on the best subset found so far (by more than `margin`).
- `backward_selection(X, y, betas, selection_cfg=None, optimizer_cfg=None)`: greedy backward selection using a shared ranking; SNPs are removed and kept only if the optimised validation AUC improves on the best subset found so far (by more than `margin`).
- `rank_snps_by_importance(X, y, betas)`: rank SNPs by their marginal AUC contribution (rank-one update trick).

### Plotting (`grsopt.draw`)

- `grs_pic(GRS, model_name, save_path=None)`: plot the GRS distribution by disease label. `GRS` is a DataFrame with columns `grs` and `Disease`.
- `roc_pic(GRS, model_name, save_path=None)`: plot the ROC curve and display AUC.

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