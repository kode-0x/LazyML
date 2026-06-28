"""Hyperparameter search spaces for Optuna tuning."""

from typing import Any


def get_search_space(model_name: str, task_type: str) -> dict[str, Any]:
    """Get hyperparameter search space for a model."""
    spaces = {
        "RandomForest": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "ExtraTrees": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "GradientBoosting": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "AdaBoost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "learning_rate": {"type": "float", "low": 0.01, "high": 2.0, "log": True},
        },
        "DecisionTree": {
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "XGBoost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "gamma": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "LightGBM": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 150},
            "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "bagging_freq": {"type": "int", "low": 1, "high": 10},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "CatBoost": {
            "iterations": {"type": "int", "low": 50, "high": 500, "log": True},
            "depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        "LogisticRegression": {
            "C": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
            "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
            "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
        },
        "Ridge": {
            "alpha": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        },
        "Lasso": {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        },
        "KNN": {
            "n_neighbors": {"type": "int", "low": 3, "high": 30},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "metric": {"type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]},
        },
        "SVM": {
            "C": {"type": "float", "low": 1e-3, "high": 1000.0, "log": True},
            "kernel": {"type": "categorical", "choices": ["rbf", "poly", "sigmoid"]},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        },
        "GaussianNB": {},  # No hyperparameters to tune for GaussianNB
    }

    return spaces.get(model_name, {})


def create_model_with_params(model_name: str, task_type: str, params: dict[str, Any]) -> Any:
    """Create a model instance with given parameters."""
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # Handle conditional parameters
    if model_name == "LogisticRegression":
        if params.get("penalty") == "l1":
            params["solver"] = "saga"
            params.pop("l1_ratio", None)
        elif params.get("penalty") == "elasticnet":
            params["solver"] = "saga"
            params["l1_ratio"] = 0.5
        elif params.get("penalty") == "l2":
            params.pop("l1_ratio", None)

    model_map = {
        "RandomForest": RandomForestClassifier if task_type == "classification" else RandomForestRegressor,
        "ExtraTrees": ExtraTreesClassifier if task_type == "classification" else ExtraTreesRegressor,
        "GradientBoosting": GradientBoostingClassifier if task_type == "classification" else GradientBoostingRegressor,
        "AdaBoost": AdaBoostClassifier if task_type == "classification" else AdaBoostRegressor,
        "DecisionTree": DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor,
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "KNN": KNeighborsClassifier,
        "SVM": SVC,
        "GaussianNB": GaussianNB,
    }

    # Optional models
    try:
        from xgboost import XGBClassifier, XGBRegressor
        model_map["XGBoost"] = XGBClassifier if task_type == "classification" else XGBRegressor
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        model_map["LightGBM"] = LGBMClassifier if task_type == "classification" else LGBMRegressor
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        model_map["CatBoost"] = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    except ImportError:
        pass

    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Set default random_state where applicable
    if "random_state" not in params and hasattr(model_class, "random_state"):
        params["random_state"] = 42
    if "random_seed" not in params and hasattr(model_class, "random_seed"):
        params["random_seed"] = 42

    return model_class(**params)
