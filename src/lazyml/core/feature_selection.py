"""Feature selection algorithms."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)


class FeatureSelector:
    """Select optimal features for training."""

    def __init__(
        self,
        task_type: str,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        use_model_selection: bool = False,
        max_features: int | None = None,
    ) -> None:
        self.task_type = task_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.use_model_selection = use_model_selection
        self.max_features = max_features
        self.selected_features: list[str] = []
        self.feature_scores: dict[str, float] = {}
        self._is_fitted = False

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
        """Apply feature selection and return transformed data."""
        df = X.copy()
        original_features = list(df.columns)

        # Step 1: Remove low variance features
        if self.variance_threshold > 0:
            df = self._remove_low_variance(df)

        # Step 2: Remove highly correlated features
        df = self._remove_correlated(df)

        # Step 3: Model-based selection
        if self.use_model_selection or self.max_features is not None:
            df = self._model_selection(df, y)

        self.selected_features = list(df.columns)
        self._is_fitted = True
        removed = set(original_features) - set(self.selected_features)
        if removed:
            print(f"  Feature selection: removed {len(removed)} features ({', '.join(removed)})")

        return df, self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the selection learned in ``fit_transform`` to new data.

        No statistics are recomputed; the exact columns kept during fit are
        selected and returned in the same order, so test/inference data is
        processed identically to training data.
        """
        if not self._is_fitted:
            raise ValueError("FeatureSelector must be fit before transform")
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing selected feature columns: {missing}")
        return X[self.selected_features]

    def _remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with variance below threshold."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df

        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(df[numeric_cols])

        mask = selector.get_support()
        kept_cols = numeric_cols[mask]
        dropped_cols = set(numeric_cols) - set(kept_cols)

        if dropped_cols:
            print(f"  Low variance: dropped {len(dropped_cols)} features")

        # Keep non-numeric columns as-is
        non_numeric = df.drop(columns=numeric_cols, errors="ignore")
        return pd.concat([df[kept_cols], non_numeric], axis=1)

    def _remove_correlated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            correlated = upper[col][upper[col] > self.correlation_threshold].index.tolist()
            to_drop.update(correlated)

        if to_drop:
            print(f"  High correlation: dropped {len(to_drop)} features")

        non_numeric = df.drop(columns=numeric_cols, errors="ignore")
        kept_numeric = [c for c in numeric_cols if c not in to_drop]
        return pd.concat([df[kept_numeric], non_numeric], axis=1)

    def _model_selection(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Use model importance for feature selection."""
        # Only numeric features for this step
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return df

        # Select k best features based on univariate tests
        k = self.max_features or min(50, len(numeric_df.columns))
        k = min(k, len(numeric_df.columns))

        score_func = f_classif if self.task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)

        selector.fit(numeric_df, y)
        mask = selector.get_support()
        selected_cols = numeric_df.columns[mask].tolist()

        # Store scores
        scores = selector.scores_[mask]
        self.feature_scores = dict(zip(selected_cols, scores, strict=False))

        non_numeric = df.drop(columns=numeric_df.columns, errors="ignore")
        return pd.concat([df[selected_cols], non_numeric], axis=1)

    def get_feature_importance_report(self) -> dict[str, Any]:
        """Get feature importance report."""
        return {
            "selected_features": self.selected_features,
            "n_selected": len(self.selected_features),
            "feature_scores": self.feature_scores,
        }
