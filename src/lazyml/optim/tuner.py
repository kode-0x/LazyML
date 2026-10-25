"""Optuna-based hyperparameter tuning (optional)."""

from collections.abc import Callable
from typing import Any

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np
from sklearn.model_selection import cross_val_score


class OptunaTuner:
    """Hyperparameter tuning with Optuna."""

    def __init__(self, n_trials: int = 50, timeout: int | None = None) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.n_trials = n_trials
        self.timeout = timeout
        self.study: optuna.Study | None = None

    def tune(
        self,
        model_factory: Callable[[dict[str, Any]], Any],
        param_distributions: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy",
        cv: int = 5,
    ) -> tuple[Any, dict[str, Any]]:
        """Tune hyperparameters and return best model."""

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for name, dist in param_distributions.items():
                if dist["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, dist["low"], dist["high"], log=dist.get("log", False)
                    )
                elif dist["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, dist["low"], dist["high"], log=dist.get("log", False)
                    )
                elif dist["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, dist["choices"])

            # Create model and evaluate
            model = model_factory(params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        # Create and run study
        direction = "maximize" if scoring in ["accuracy", "r2", "f1"] else "minimize"
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(
            objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True
        )

        # Train final model with best params
        best_params = self.study.best_params
        best_model = model_factory(best_params)
        best_model.fit(X, y)

        return best_model, best_params

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history."""
        if self.study is None:
            return []

        return [
            {"trial": t.number, "value": t.value, "params": t.params}
            for t in self.study.trials
            if t.value is not None
        ]
