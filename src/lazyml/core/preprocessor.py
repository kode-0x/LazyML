"""Data preprocessing pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class DataPreprocessor:
    """Preprocess data for machine learning."""

    def __init__(self) -> None:
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] = []
        self.target_encoder: LabelEncoder | None = None
        self._is_fitted = False

    def fit_transform(
        self, df: pd.DataFrame, target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data."""
        X = df.drop(columns=[target_column])
        y = df[target_column].values

        # Encode target if classification
        if self._is_classification_target(y):
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Build preprocessing pipeline
        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self._get_feature_names(X)
        self._is_fitted = True

        return X_processed, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self._is_fitted or self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform")

        return self.preprocessor.transform(df)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded target back to original labels."""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y)
        return y

    def _get_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Get processed feature names."""
        if self.preprocessor is None:
            return []

        names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(columns))
            else:
                names.extend(columns)
        return names

    @staticmethod
    def _is_classification_target(y: np.ndarray) -> bool:
        """Check if target is for classification."""
        if y.dtype == object or hasattr(y, "dtype") and str(y.dtype) == "object":
            return True
        unique = np.unique(y[~pd.isna(y)])
        return len(unique) <= 10 if len(unique) < len(y) * 0.1 else False
