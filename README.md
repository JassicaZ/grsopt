# grsopt

A Python package for optimizing genetic risk score (GRS) models. `grsopt` supports CV-based forward/backward SNP selection, beta optimization for genetic risk scores, and visualization of GRS distributions and ROC curves.

## Key Features

- Optimize published GRS weights using gradient-based methods
- CV-based **forward** or **backward** SNP feature selection 
- Plot GRS score distributions and ROC curves

## Installation

Install from source:

```bash
git clone https://github.com/JassicaZ/grsopt.git
cd grsopt
pip install .
```

## Quick Start

### Feature selection
```python
import grsopt
from grsopt import SelectionConfig, OptimizerConfig

# X: genotype DataFrame (individuals x SNPs), y: binary labels (0/1)
# betas: original GRS coefficients indexed by SNP name

cfg = SelectionConfig(
    direction="forward", cv_folds=3, start_n=10, end_n=None,
    margin=0.0, retention_threshold=1,
)
opt = OptimizerConfig(max_iter=500, n_initial=3)

result = grsopt.forward_selection(X, y, betas, cfg, opt)
print(result["selected_snps"])   # final selected SNP list
print(result["mean_auc"])        # cross-fold validation AUC (mildly optimistic)
```

### Beta optimization
```python
from grsopt import grsoptimizer

opt = grsoptimizer(
    original_betas=betas,
    regularization_strength=0.1,
    max_iter=200, n_initial=5, noise_scale=0.05,
    verbose=True
)
result = opt.optimize(X, y)
```

## Method

### GRS scoring

The score of individual $i$ is the weighted sum of its genotype dosages:

$$\mathrm{GRS}_i=\sum_{j=1}^{p}X_{ij}\beta_j$$

where $X_{ij}\in\{0,1,2\}$ is the dosage of SNP $j$ and $\beta_j$ its weight. AUC is computed from these scores.

### Beta optimization

The optimizer keeps this score form and tunes the weights around the published ones. For the current SNP set $S$:

$$J(\boldsymbol\beta)=\mathcal{L}(\boldsymbol\beta)+\lambda\sum_{j\in S}\left(\beta_j-\beta_j^{(0)}\right)^2$$

$\beta_j^{(0)}$ is the published weight of SNP $j$, $\lambda$ is `regularization_strength`, and the optimizer minimizes $J$. The data term $\mathcal{L}$ follows from `loss`:

- `log`: $\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\sigma(s_i)+(1-y_i)\log(1-\sigma(s_i))\right]$
- `rank`: $\mathcal{L}=\frac{1}{|P||N|}\sum_{i\in P}\sum_{j\in N}\log\left(1+e^{-(s_i-s_j)}\right)$

with $s_i=\mathrm{GRS}_i$, $\sigma$ the sigmoid, and $P$ / $N$ the case / control sets. If no initialization beats the published betas on training AUC, the published betas are returned unchanged.

### SNP selection

Selection is a greedy search over SNP subsets, repeated inside each cross-validation fold.

1. **Rank the candidates.** Score each SNP by how much AUC drops when it is removed from the current set, then order the SNPs from most to least useful.
2. **Forward/backward selection.** Forward selection walks the ranking from the top and tries adding one SNP; backward selection walks it from the bottom and tries removing one. A trial subset $S'$ is accepted only if its optimized validation AUC beats the best value found so far in that fold, $\mathrm{AUC}^{\ast}$, by more than `margin`: $\mathrm{AUC}_{\mathrm{val}}(S')-\mathrm{AUC}^{\ast}>\mathrm{margin}$. The search stops when no single change is accepted.
3. **Count across folds.** A SNP is kept in the final signature when it is selected in at least a `retention_threshold` fraction of the $K$ folds.

Forward starts from the top `start_n` SNPs and grows; backward starts from all SNPs and shrinks (see `SelectionConfig`). 

## API

### Configuration (`grsopt.config`)

- `OptimizerConfig`: `regularization_strength`, `max_iter`, `n_initial`, `noise_scale`, `loss`, `verbose`
- `SelectionConfig`: `direction` (`'forward'`/`'backward'`), `cv_folds`, `random_state`, `start_n`, `end_n`, `margin`, `retention_threshold`

### Beta optimization (`grsopt.grsoptimizer`)

Class for optimizing GRS beta weights.

#### Parameters

- `original_betas`: array-like, original published GRS coefficients in the same column order as `X`.
- `regularization_strength`: float, regularization weight to control deviation from original betas
- `max_iter`: int, maximum number of optimization iterations
- `n_initial`: int, number of random initializations to reduce local optimum risk
- `noise_scale`: float, scale of random perturbation for initial beta values
- `loss`: `'log'` or `'rank'`, default `'rank'`.
    - `'log'`: logistic cross-entropy based on sigmoid disease probabilities; it focuses on probabilistic classification.
    - `'rank'`: pairwise logistic ranking loss comparing positive-negative sample pairs; it focuses on the ordering measured by AUC.
- `verbose`: bool, default `False`. If `True`, print progress messages during optimisation and cross-validation.

#### Methods

- `optimize(X, y)`: optimize betas on training data and return a result dictionary. `X` is a genotype matrix (rows = samples, columns = SNPs, encoded as `0/1/2`).
- `cross_validate(X, y, param_grid, cv_folds=5, random_state=42)`: evaluate hyperparameter combinations using stratified CV.

### Scoring (`grsopt.metrics`)

- `calculate_auc(X, y, betas=None, selected_features='all')`: AUC of a GRS built from a subset of SNPs. `betas` is **required** — passing `None` raises a `ValueError` (an unweighted/equal-weight GRS is no longer implicit).

### Feature selection (`grsopt.selection`)

Both return a dict with `per_fold`, `frequency`, `selected_snps`, `mean_auc` and `std_auc`. During selection they print the per-fold add/remove decision with the validation-AUC change, and finish with `Final selected SNPs count`.

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

- 🔒 **No example genotype data** is provided here due to privacy and ethical restrictions.
- 🧪 The package has currently been validated on our internal datasets. If you test it on independent datasets, we would greatly appreciate your feedback.
- 💬 If you encounter any issues, unexpected results, or are interested in collaboration, please open an Issue or contact us.