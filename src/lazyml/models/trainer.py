"""Model training with multiple algorithms."""

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def _train_single_model(
    name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    scoring: str,
) -> tuple[str, dict[str, Any]]:
    """Train a single model - helper for parallel execution."""
    try:
        model_start = time.time()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean_score = scores.mean()

        if scoring == "neg_root_mean_squared_error":
            mean_score = -mean_score

        model.fit(X, y)
        train_time = time.time() - model_start

        return name, {
            "model": model,
            "cv_score": mean_score,
            "cv_std": scores.std(),
            "train_time": train_time,
            "fitted": True,
        }
    except Exception as e:
        return name, {
            "model": model,
            "cv_score": 0.0 if "classification" in str(type(model)) else float("inf"),
            "cv_std": 0.0,
            "train_time": 0.0,
            "fitted": False,
            "error": str(e),
        }


class ModelTrainer:
    """Train multiple models with cross-validation."""

    def __init__(
        self,
        task_type: str,
        cv_folds: int = 5,
        time_budget: int = 300,
        ensemble: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        use_optuna: bool = False,
        optuna_trials: int = 50,
        optuna_timeout: int | None = None,
        early_stopping: bool = False,
    ) -> None:
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.time_budget = time_budget
        self.ensemble = ensemble
        self.parallel = parallel
        self.max_workers = max_workers
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        self.early_stopping = early_stopping
        self.models: dict[str, Any] = {}
        self.best_params: dict[str, dict[str, Any]] = {}
        self._setup_models()

    def _setup_models(self) -> None:
        """Initialize model configurations."""
        if self.task_type == "classification":
            self.models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "SVM": SVC(random_state=42, probability=True),
                "GaussianNB": GaussianNB(),
            }
        else:
            self.models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42, max_iter=1000),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
            }

        # Try optional models
        try:
            from xgboost import XGBClassifier, XGBRegressor

            if self.task_type == "classification":
                self.models["XGBoost"] = XGBClassifier(
                    n_estimators=100, random_state=42, verbosity=0
                )
            else:
                self.models["XGBoost"] = XGBRegressor(
                    n_estimators=100, random_state=42, verbosity=0
                )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            if self.task_type == "classification":
                self.models["LightGBM"] = LGBMClassifier(
                    n_estimators=100, random_state=42, verbosity=-1
                )
            else:
                self.models["LightGBM"] = LGBMRegressor(
                    n_estimators=100, random_state=42, verbosity=-1
                )
        except ImportError:
            pass

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            if self.task_type == "classification":
                self.models["CatBoost"] = CatBoostClassifier(
                    iterations=100, random_seed=42, verbose=False
                )
            else:
                self.models["CatBoost"] = CatBoostRegressor(
                    iterations=100, random_seed=42, verbose=False
                )
        except ImportError:
            pass

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, Any]]:
        """Train all models and return results."""
        results = {}
        start_time = time.time()

        cv = self._get_cv_splitter()
        scoring = (
            "accuracy" if self.task_type == "classification" else "neg_root_mean_squared_error"
        )

        # Use Optuna for hyperparameter tuning if enabled
        if self.use_optuna and OPTUNA_AVAILABLE:
            print(
                f"Training with Optuna hyperparameter tuning ({self.optuna_trials} trials per model)..."
            )
            results = self._train_with_optuna(X, y, cv, scoring, start_time)
        elif self.parallel and len(self.models) > 1:
            print(f"Training {len(self.models)} models in parallel...")
            results = self._train_parallel(X, y, cv, scoring, start_time)
        else:
            results = self._train_sequential(X, y, cv, scoring, start_time)

        # Build ensemble if requested and we have enough models
        if self.ensemble and len(results) >= 2:
            self._build_and_train_ensemble(results, X, y, cv, scoring, start_time)

        return results

    def _build_and_train_ensemble(
        self,
        results: dict[str, dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        scoring: str,
        start_time: float,
    ) -> None:
        """Build and train ensemble from best models."""
        from lazyml.models.ensemble import create_ensemble_from_results

        # Determine ensemble type based on task
        ensemble_type = "voting" if self.task_type == "classification" else "stacking"

        print(f"\nBuilding {ensemble_type} ensemble from top models...")

        ensemble_result = create_ensemble_from_results(
            results, self.task_type, ensemble_type=ensemble_type, top_n=3
        )

        if ensemble_result is None:
            print("  Could not build ensemble: insufficient fitted models")
            return

        ensemble, model_names = ensemble_result

        # Check if we have time to train ensemble
        if self.time_budget and (time.time() - start_time) > self.time_budget * 0.9:
            print("  Skipping ensemble: time budget nearly exhausted")
            return

        try:
            # Train the ensemble
            ensemble.fit(X, y)

            # Evaluate ensemble
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(ensemble, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = scores.mean()

            if scoring == "neg_root_mean_squared_error":
                mean_score = -mean_score

            # Store ensemble in results
            ensemble_name = f"Ensemble ({ensemble_type})"
            results[ensemble_name] = {
                "model": ensemble,
                "cv_score": mean_score,
                "cv_std": scores.std(),
                "train_time": 0.0,  # Included in total training time
                "fitted": True,
                "is_ensemble": True,
                "component_models": model_names,
            }

            print(f"  Ensemble score: {mean_score:.4f} (from {', '.join(model_names)})")

        except Exception as e:
            print(f"  Ensemble training failed: {e}")

    def _train_with_optuna(
        self, X: np.ndarray, y: np.ndarray, cv: Any, scoring: str, start_time: float
    ) -> dict[str, dict[str, Any]]:
        """Train models with Optuna hyperparameter tuning."""
        results = {}

        # Calculate time budget per model
        n_models = len(self.models)
        time_per_model = (self.time_budget / n_models) if self.time_budget else None

        for name in self.models:
            if time.time() - start_time > self.time_budget:
                print("Time budget reached. Stopping training.")
                break

            print(f"Tuning {name} with Optuna...")
            model_start = time.time()

            try:
                # Calculate remaining time for this model
                remaining = (
                    self.time_budget - (time.time() - start_time) if self.time_budget else None
                )
                if remaining and remaining < 30:  # Skip if less than 30s remaining
                    print(f"  Skipping {name}: insufficient time remaining")
                    continue

                if remaining is not None and time_per_model is not None:
                    model_timeout: float | None = min(time_per_model, remaining)
                else:
                    model_timeout = time_per_model

                best_model, best_params, best_score = self._tune_model_with_optuna(
                    name, X, y, cv, scoring, timeout=model_timeout
                )
                train_time = time.time() - model_start

                self.best_params[name] = best_params

                results[name] = {
                    "model": best_model,
                    "cv_score": best_score,
                    "cv_std": 0.0,  # Optuna reports single best score
                    "train_time": train_time,
                    "fitted": True,
                    "best_params": best_params,
                }

                params_str = ", ".join(
                    [
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in list(best_params.items())[:3]
                    ]
                )
                print(f"  Score: {best_score:.4f} in {train_time:.2f}s (params: {params_str}...)")

            except Exception as e:
                print(f"  Failed: {e}")
                results[name] = {
                    "model": self.models[name],
                    "cv_score": 0.0 if self.task_type == "classification" else float("inf"),
                    "cv_std": 0.0,
                    "train_time": 0.0,
                    "fitted": False,
                    "error": str(e),
                }

        return results

    def _train_sequential(
        self, X: np.ndarray, y: np.ndarray, cv: Any, scoring: str, start_time: float
    ) -> dict[str, dict[str, Any]]:
        """Train models sequentially."""
        results = {}

        for name, model in self.models.items():
            if time.time() - start_time > self.time_budget:
                print("Time budget reached. Stopping training.")
                break

            print(f"Training {name}...")
            model_start = time.time()

            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()

                if scoring == "neg_root_mean_squared_error":
                    mean_score = -mean_score

                model.fit(X, y)
                train_time = time.time() - model_start

                results[name] = {
                    "model": model,
                    "cv_score": mean_score,
                    "cv_std": scores.std(),
                    "train_time": train_time,
                    "fitted": True,
                }

                print(f"  Score: {mean_score:.4f} (+/- {scores.std():.4f}) in {train_time:.2f}s")

            except Exception as e:
                print(f"  Failed: {e}")
                results[name] = {
                    "model": model,
                    "cv_score": 0.0 if self.task_type == "classification" else float("inf"),
                    "cv_std": 0.0,
                    "train_time": 0.0,
                    "fitted": False,
                    "error": str(e),
                }

        return results

    def _train_parallel(
        self, X: np.ndarray, y: np.ndarray, cv: Any, scoring: str, start_time: float
    ) -> dict[str, dict[str, Any]]:
        """Train models in parallel using ProcessPoolExecutor."""
        results = {}
        workers = self.max_workers or min(len(self.models), 4)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(_train_single_model, name, model, X, y, cv, scoring): name
                for name, model in self.models.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                if time.time() - start_time > self.time_budget:
                    print("Time budget reached. Stopping training.")
                    break

                name = futures[future]
                try:
                    model_name, result = future.result()
                    results[model_name] = result

                    if result["fitted"]:
                        print(
                            f"  {model_name}: {result['cv_score']:.4f} in {result['train_time']:.2f}s"
                        )
                    else:
                        print(f"  {model_name}: Failed - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"  {name}: Failed - {e}")
                    results[name] = {
                        "model": self.models[name],
                        "cv_score": 0.0 if self.task_type == "classification" else float("inf"),
                        "cv_std": 0.0,
                        "train_time": 0.0,
                        "fitted": False,
                        "error": str(e),
                    }

        return results

    def _get_cv_splitter(self) -> StratifiedKFold | KFold:
        """Get cross-validation splitter based on task type."""
        if self.task_type == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def _tune_model_with_optuna(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        scoring: str,
        timeout: float | None = None,
    ) -> tuple[Any, dict[str, Any], float]:
        """Tune a single model using Optuna."""
        from lazyml.optim.search_spaces import create_model_with_params, get_search_space

        search_space = get_search_space(model_name, self.task_type)
        if not search_space:
            # No search space defined: evaluate the default model and report
            # its actual CV score so it ranks fairly against tuned models.
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            score = scores.mean()
            if scoring == "neg_root_mean_squared_error":
                score = -score
            model.fit(X, y)
            return model, {}, float(score)

        direction = "maximize" if scoring in ["accuracy", "r2", "f1"] else "minimize"

        # Create study with pruner for early stopping
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        sampler = TPESampler(seed=42)

        study = optuna.create_study(direction=direction, pruner=pruner, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters from search space
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Create and evaluate model
            try:
                model = create_model_with_params(model_name, self.task_type, params)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                score = scores.mean()

                # Report intermediate result for pruning
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score
            except optuna.TrialPruned:
                raise
            except Exception as err:
                # A parameter combination that fails to fit is treated as a
                # pruned trial rather than aborting the whole study.
                raise optuna.TrialPruned() from err

        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,  # We parallelize across models, not within Optuna
        )

        # Train final model with best params
        best_params = study.best_params
        best_model = create_model_with_params(model_name, self.task_type, best_params)
        best_model.fit(X, y)

        return best_model, best_params, study.best_value
