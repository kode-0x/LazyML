"""Model selection and ranking."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
)


class ModelSelector:
    """Select best model from trained candidates."""

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type

    def select_best(
        self, results: dict[str, dict[str, Any]], X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Select best model and create leaderboard."""
        leaderboard = []

        for name, result in results.items():
            if not result.get("fitted", False):
                continue

            model = result["model"]
            cv_score = result["cv_score"]
            train_time = result["train_time"]

            # Evaluate on test set
            try:
                predictions = model.predict(X_test)
                test_metrics = self._calculate_metrics(y_test, predictions)
            except Exception:
                test_metrics = {}

            entry = {
                "model": name,
                "cv_score": cv_score,
                "cv_std": result.get("cv_std", 0.0),
                "train_time": train_time,
                **test_metrics,
            }
            leaderboard.append(entry)

        # Sort leaderboard
        if self.task_type == "classification":
            leaderboard.sort(key=lambda x: x.get("accuracy", 0.0), reverse=True)
        else:
            # For regression, sort by R² (higher is better)
            leaderboard.sort(key=lambda x: x.get("r2", float("-inf")), reverse=True)

        # Get best model
        if not leaderboard:
            raise ValueError("No models were successfully trained")

        best_name = leaderboard[0]["model"]
        best_model = results[best_name]["model"]

        # Print leaderboard
        print("\n" + "=" * 50)
        print("MODEL LEADERBOARD")
        print("=" * 50)
        for i, entry in enumerate(leaderboard, 1):
            if self.task_type == "classification":
                print(f"{i}. {entry['model']:<15} - Accuracy: {entry.get('accuracy', 0):.4f}")
            else:
                print(f"{i}. {entry['model']:<15} - R²: {entry.get('r2', 0):.4f}, RMSE: {entry.get('rmse', 0):.4f}")
        print("=" * 50)

        return best_model, leaderboard

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate appropriate metrics based on task type."""
        metrics = {}

        if self.task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            # Use weighted average for multi-class
            if len(np.unique(y_true)) > 2:
                metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            else:
                metrics["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
        else:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        return metrics
