Optimization Module (lazyml.optim)
====================================

The ``lazyml.optim`` package provides hyperparameter optimization capabilities.

OptunaTuner
-----------

.. autoclass:: lazyml.optim.tuner.OptunaTuner
   :members:
   :undoc-members:
   :show-inheritance:

Search Spaces
-------------

.. autofunction:: lazyml.optim.search_spaces.get_search_space

.. autofunction:: lazyml.optim.search_spaces.create_model_with_params

Hyperparameter Spaces
----------------------

The following models have predefined hyperparameter search spaces:

- **RandomForest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **ExtraTrees**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **GradientBoosting**: n_estimators, max_depth, learning_rate, subsample
- **AdaBoost**: n_estimators, learning_rate
- **DecisionTree**: max_depth, min_samples_split, min_samples_leaf
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
- **LightGBM**: n_estimators, max_depth, learning_rate, num_leaves, feature_fraction, bagging_fraction, bagging_freq, min_child_samples, reg_alpha, reg_lambda
- **CatBoost**: iterations, depth, learning_rate, l2_leaf_reg
- **LogisticRegression**: C, penalty, solver, class_weight
- **Ridge**: alpha
- **Lasso**: alpha
- **KNN**: n_neighbors, weights, metric
- **SVM**: C, kernel, gamma

Optimization Strategy
----------------------

The tuner uses:

- **TPE Sampler** - Tree-structured Parzen Estimator for Bayesian optimization
- **Median Pruner** - Early stopping for underperforming trials
- **Time Budgeting** - Distributes optimization time across models
- **Cross-Validation** - Evaluates each configuration with K-fold CV

Usage
-----

.. code-block:: python

   from lazyml.optim import OptunaTuner

   tuner = OptunaTuner(n_trials=50, timeout=300)
   best_model, best_params = tuner.tune(
       model_factory=lambda p: RandomForestClassifier(**p),
       param_distributions=search_space,
       X=X_train,
       y=y_train,
       scoring="accuracy",
       cv=5,
   )
