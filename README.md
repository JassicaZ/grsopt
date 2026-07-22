# grsopt

A lightweight Python package for optimizing genetic risk score (GRS) models. `grsopt` supports beta optimization for genetic risk scores (GRS) model and visualization of GRS distributions and ROC curves.

## Key Features

- Optimize published GRS weights using gradient-based methods
- Support cross-validation over regularization and noise parameters
- Plot GRS score distributions and ROC curves

## Installation

Install from source:

```bash
git clone https://github.com/JassicaZ/grsopt.git
cd grsopt
pip install .
```
## Quick Start

See `demo.ipynb` for a step-by-step example demonstrating the usage of the package.

## API

### `grsopt.grsoptimizer`

Class for optimizing GRS beta weights.

#### Parameters

- `original_betas`: array-like, original published GRS coefficients in the same column order as `X`.
- `regularization_strength`: float, regularization weight to control deviation from original betas
- `max_iter`: int, maximum number of optimization iterations
- `n_intial`: int, number of random initializations to reduce local optimum risk
- `noise_scale`: float, scale of random perturbation for initial beta values

#### Methods

- `optimize(X, y)`: optimize betas on training data and return a result dictionary. X: A genotype matrix where rows represent samples and columns represent SNPs. Genotypes should be encoded as `0`, `1`, or `2`. 
- `cross_validate(X, y, param_grid, cv_folds=5)`: evaluate hyperparameter combinations using stratified CV

### Plotting utilities

- `grsopt.GRS_pic(GRS, model_name, save_path=None)`: plot the GRS distribution by disease label.
  - `GRS`: DataFrame with columns `grs` (with calculated GRS scores for each individual) and `Disease`.
  - `model_name`: string identifier used for the plot and saved file name.
  - `save_path`: optional directory path to save the displayed image.Default is `None` (the image is not saved).
- `grsopt.ROC_pic(GRS, model_name, save_path=None)`: plot the ROC curve and display AUC.
  - `GRS`: DataFrame with columns `grs` (with calculated GRS scores for each individual)and `Disease`.
  - `model_name`: string identifier used for the plot and saved file name.
  - `save_path`: optional directory path to save the displayed image.Default is `None` (the image is not saved).

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