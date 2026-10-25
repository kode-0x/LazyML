"""Handle imbalanced datasets."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


class ImbalanceHandler:
    """Detect and handle class imbalance."""

    def __init__(self, threshold: float = 0.2) -> None:
        """
        Initialize imbalance handler.

        Args:
            threshold: Minority class ratio below which imbalance is flagged
        """
        self.threshold = threshold
        self.is_imbalanced = False
        self.imbalance_ratio = 1.0
        self.original_distribution: dict[Any, int] = {}

    def analyze(self, y: np.ndarray) -> dict[str, Any]:
        """Analyze class distribution."""
        counts = Counter(y)
        self.original_distribution = dict(counts)

        min_count = min(counts.values())
        max_count = max(counts.values())

        self.imbalance_ratio = min_count / max_count if max_count > 0 else 1.0
        self.is_imbalanced = self.imbalance_ratio < self.threshold

        return {
            "is_imbalanced": self.is_imbalanced,
            "imbalance_ratio": self.imbalance_ratio,
            "distribution": dict(counts),
            "minority_class": min(counts, key=lambda c: counts[c]),
            "majority_class": max(counts, key=lambda c: counts[c]),
        }

    def get_class_weights(self, y: np.ndarray) -> dict[int, float] | None:
        """Calculate balanced class weights."""
        if not self.is_imbalanced:
            return None

        counts = Counter(y)
        total = len(y)
        n_classes = len(counts)

        # Balanced weight: n_samples / (n_classes * n_samples_for_class)
        weights = {cls: total / (n_classes * count) for cls, count in counts.items()}

        return weights

    def resample(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "auto"
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Resample data to handle imbalance.

        Args:
            X: Feature dataframe
            y: Target array
            method: 'smote', 'undersample', or 'auto'

        Returns:
            Resampled X and y
        """
        if not self.is_imbalanced:
            return X, y

        if method == "auto":
            method = "smote" if self.imbalance_ratio > 0.1 else "smote"

        print(f"  Handling imbalance with {method} (ratio: {self.imbalance_ratio:.2f})")

        if method == "smote":
            return self._apply_smote(X, y)

        return X, y

    def resample_array(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resample an already-encoded numeric feature matrix with SMOTE.

        Intended for use on the *training split only*, after preprocessing has
        produced a dense numeric array. If SMOTE is unavailable or fails, the
        inputs are returned unchanged so training can proceed.
        """
        if not self.is_imbalanced:
            return X, y
        try:
            from imblearn.over_sampling import SMOTE

            min_count = min(Counter(y).values())
            if min_count < 2:
                print("  Skipping SMOTE: a class has fewer than 2 samples")
                return X, y
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
            X_res, y_res = smote.fit_resample(X, y)
            print(f"  SMOTE: {len(y)} -> {len(y_res)} samples")
            return X_res, y_res
        except ImportError:
            print("  imbalanced-learn not installed. Install with: uv pip install imbalanced-learn")
            return X, y
        except Exception as e:  # pragma: no cover - defensive
            print(f"  SMOTE failed: {e}")
            return X, y

    def _apply_smote(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Apply SMOTE for oversampling minority class."""
        try:
            from imblearn.over_sampling import SMOTE, SMOTENC

            # Check if we have categorical features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_indices = [i for i, col in enumerate(X.columns) if col not in numeric_cols]

            if categorical_indices and len(categorical_indices) < len(X.columns):
                # Mixed types - use SMOTENC
                smote = SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=42,
                    k_neighbors=min(5, min(Counter(y).values()) - 1),
                )
            else:
                # All numeric - use SMOTE
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=min(5, min(Counter(y).values()) - 1),
                )

            X_resampled, y_resampled = smote.fit_resample(X, y)

            print(f"  SMOTE: {len(y)} → {len(y_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), y_resampled

        except ImportError:
            print("  imbalanced-learn not installed. Install with: uv pip install imbalanced-learn")
            return X, y
        except Exception as e:
            print(f"  SMOTE failed: {e}")
            return X, y
