# LazyML

LazyML command-line tool for automated machine learning on CSV datasets.

## Overview

LazyML automates the end-to-end machine learning workflow:

1. **Data Profiling** - Automatic schema detection, missing value analysis, feature type identification
2. **Preprocessing** - Handles missing values, encoding, scaling automatically
3. **Model Training** - Trains multiple algorithms with cross-validation
4. **Hyperparameter Optimization** - Bayesian optimization via Optuna
5. **Model Selection** - Ranks models by performance metrics
6. **Export** - Saves best model for deployment

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/username/lazyml.git
cd lazyml
uv sync
```

### Using pip

```bash
git clone https://github.com/username/lazyml.git
cd lazyml
pip install -e .
```

### Optional Dependencies

For enhanced model support (XGBoost, LightGBM, CatBoost):

```bash
uv sync --extra boost
```

For SMOTE class imbalance handling:

```bash
uv sync --extra imbalanced
```

For hyperparameter optimization:

```bash
uv sync --extra optuna
```

For TUI (Terminal User Interface):

```bash
uv sync --extra tui
```

Install all extras at once:

```bash
uv sync --all-extras
```

## Quick Start

### Train on a CSV file

```bash
uv run python -m lazyml run data.csv --target target_column
```

Or using the installed command:

```bash
uv run lazyml run data.csv --target target_column
```

### Interactive Mode

```bash
uv run lazyml run
```

### Make Predictions

```bash
uv run lazyml predict model.pkl new_data.csv --output predictions.csv
```

## Usage

### Command Reference

| Command                  | Description             |
| ------------------------ | ----------------------- |
| `run [file]`             | Train models on dataset |
| `predict <model> <data>` | Generate predictions    |
| `profile <file>`         | Analyze dataset only    |

### Training Options

```bash
uv run lazyml run data.csv \
    --target price \
    --task regression \
    --time-budget 600 \
    --cv-folds 5 \
    --output-dir ./models \
    --optuna \
    --optuna-trials 30
```

| Option                | Description                         | Default       |
| --------------------- | ----------------------------------- | ------------- |
| `--target`            | Target column name                  | (auto-detect) |
| `--task`              | `classification` or `regression`    | (auto-detect) |
| `--time-budget`       | Max training time in seconds        | 300           |
| `--cv-folds`          | Cross-validation folds              | 5             |
| `--output-dir`        | Directory to save outputs           | `./output`    |
| `--ensemble`          | Enable ensemble methods             | False         |
| `--feature-selection` | Automatic feature selection         | True          |
| `--handle-imbalance`  | Handle class imbalance (SMOTE)      | True          |
| `--parallel`          | Train models in parallel            | True          |
| `--max-workers`       | Max parallel workers                | auto          |
| `--optuna`            | Enable Optuna hyperparameter tuning | False         |
| `--optuna-trials`     | Number of trials per model          | 30            |

## Supported Models

### Classification

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Decision Tree
- XGBoost (optional)
- LightGBM (optional)
- CatBoost (optional)
- SVM
- K-Nearest Neighbors
- Gaussian Naive Bayes

### Regression

- Linear Regression
- Ridge / Lasso
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Decision Tree
- XGBoost (optional)
- LightGBM (optional)
- CatBoost (optional)

## Output Structure

```
output/
├── model.pkl              # Best trained model
├── preprocessor.pkl       # Fitted preprocessing pipeline
├── report.html            # Evaluation report
├── leaderboard.json       # Model rankings
└── predictions.csv        # Validation predictions
```

## Configuration

Create `lazyml.yaml` for persistent settings:

```yaml
models:
  include:
    - random_forest
    - xgboost
    - lightgbm
  exclude:
    - svm

optimization:
  time_budget: 600
  n_trials: 50

cross_validation:
  folds: 5
  strategy: stratified # or kfold for regression
```

## Project Structure

```
lazyml/
├── cli/               # Command-line interface
├── core/              # Data loading and preprocessing
├── models/            # Model training and selection
├── optim/             # Hyperparameter optimization
└── reports/           # Output generation
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Type checking
uv run mypy src/lazyml/

# Run the CLI
uv run python -m lazyml
```

## Documentation

Build the Sphinx documentation locally:

```bash
# Install docs dependencies
uv sync --extra docs

# Build HTML documentation (Linux/macOS)
cd docs
uv run make html

# Build HTML documentation (Windows PowerShell)
cd docs
uv run sphinx-build -M html source build

# View the documentation
open build/html/index.html  # macOS
# or
start build/html/index.html  # Windows
```

## Features

- **Automatic Feature Selection** - Removes low-variance and highly correlated features
- **Class Imbalance Handling** - Auto-detects imbalance and applies SMOTE oversampling
- **Parallel Training** - Trains multiple models concurrently for faster results
- **Ensemble Learning** - Voting classifier (soft voting) and stacking regressor with top models
- **Smart Preprocessing** - Handles missing values, encoding, and scaling automatically
- **Optuna Hyperparameter Tuning** - Bayesian optimization with early stopping and pruning
- **Model Ranking** - Cross-validated comparison with multiple metrics
- **HTML Reports** - Visual leaderboards and dataset summaries

## Requirements

- Python 3.9+
- pandas
- scikit-learn
- numpy
- Typer (CLI framework)

### Optional

- Textual (TUI framework)
- XGBoost, LightGBM, CatBoost (gradient boosting libraries)
- Optuna (hyperparameter optimization)
- imbalanced-learn (SMOTE for class imbalance)

See `requirements.txt` for full dependency list.

## License

MIT
