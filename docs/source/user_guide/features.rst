Features
========

LazyML provides a comprehensive set of features for automated machine learning.

Automatic Feature Selection
-----------------------------

LazyML automatically selects relevant features to improve model performance:

- **Low Variance Removal** - Removes features with variance below threshold (default: 0.01)
- **Correlation Filtering** - Removes highly correlated features (> 0.95 correlation)
- **Statistical Selection** - Uses univariate statistical tests (f_classif/f_regression)

Enable/disable with:

.. code-block:: bash

   uv run lazyml run data.csv --target y --feature-selection
   uv run lazyml run data.csv --target y --no-feature-selection

Class Imbalance Handling
--------------------------

For classification tasks with imbalanced classes:

- **Auto-detection** - Detects imbalance when minority class < 20%
- **SMOTE Oversampling** - Generates synthetic samples for minority class
- **Class Weights** - Automatically applies balanced class weights

Requires ``imbalanced-learn``:

.. code-block:: bash

   uv sync --extra imbalanced
   uv run lazyml run data.csv --target y --handle-imbalance

Parallel Training
-----------------

Train multiple models concurrently for faster results:

.. code-block:: bash

   uv run lazyml run data.csv --target y --parallel --max-workers 4

- Automatically distributes models across workers
- Respects time budget across all models
- Falls back to sequential if parallel fails

Hyperparameter Optimization
-----------------------------

Bayesian optimization with Optuna:

.. code-block:: bash

   uv sync --extra optuna
   uv run lazyml run data.csv --target y --optuna --optuna-trials 30

Features:

- **TPE Sampler** - Tree-structured Parzen Estimator for efficient search
- **Early Pruning** - Median pruner stops unpromising trials early
- **Per-model Budget** - Distributes time budget across models
- **Search Spaces** - Predefined for all supported algorithms

Supported Models for Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Decision Tree
- XGBoost (optional)
- LightGBM (optional)
- CatBoost (optional)
- Logistic Regression
- Ridge/Lasso
- KNN
- SVM

Ensemble Learning
-----------------

Combine top models for improved performance:

.. code-block:: bash

   uv run lazyml run data.csv --target y --ensemble

**Classification:** Soft voting classifier combining top 3 models
**Regression:** Stacking regressor with Ridge meta-learner

The ensemble is trained after individual models and added to the leaderboard.

Smart Preprocessing
-------------------

Automatic data preprocessing:

- **Missing Values** - Median imputation for numeric, mode for categorical
- **Encoding** - One-hot encoding for categoricals
- **Scaling** - StandardScaler for numeric features
- **Target Encoding** - Label encoding for classification targets

The preprocessing pipeline is saved with the model for consistent inference.

Supported Models
----------------

Classification
~~~~~~~~~~~~~~~

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

Regression
~~~~~~~~~~~

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

Output Reports
--------------

HTML report includes:

- Dataset summary (rows, columns, memory)
- Model leaderboard with CV scores
- Test set metrics
- Best model identification
- Component models for ensembles

Configuration File
------------------

Create ``lazyml.yaml`` for persistent settings:

.. code-block:: yaml

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
     strategy: stratified

Performance Tips
----------------

1. **Use time budgets** - Set ``--time-budget`` to control training duration
2. **Enable parallel training** - ``--parallel`` for faster model comparison
3. **Use feature selection** - Reduces dimensionality automatically
4. **Try ensembles** - ``--ensemble`` often improves performance
5. **Hyperparameter tuning** - ``--optuna`` for best model configuration
