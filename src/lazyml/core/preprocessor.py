"""Leakage-safe data preprocessing.

The preprocessor is fit **only** on training data. All stateful steps
(imputation, scaling, encoding) learn their parameters from the training
split and are then applied unchanged to validation/test/inference data. This
prevents test-set statistics from leaking into training.

Target encoding is handled separately from feature preprocessing so the same
fitted ``ColumnTransformer`` can transform raw inference inputs (which have no
target column).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def build_feature_transformer(X: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the given feature frame.

    Numeric columns are median-imputed then standardized; categorical/string
    columns are constant-imputed then one-hot encoded. Unknown categories seen
    at transform time are ignored, so inference never fails on new levels.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


class DataPreprocessor:
    """Fit-on-train preprocessing for features and target.

    Usage::

        pre = DataPreprocessor()
        X_train_t = pre.fit(X_train_df, y_train)          # learns from train only
        X_test_t = pre.transform(X_test_df)               # applies, no refit
        y_train_enc = pre.transform_target(y_train)
    """

    def __init__(self) -> None:
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] = []
        self.input_columns: list[str] = []
        self.target_encoder: LabelEncoder | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> np.ndarray:
        """Fit the feature transformer (and target encoder) on training data.

        Returns the transformed training features. Only ``X``/``y`` from the
        training split should ever be passed here.
        """
        y_arr = np.asarray(y)
        self.input_columns = list(X.columns)

        if self._is_classification_target(y_arr):
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y_arr)

        self.preprocessor = build_feature_transformer(X)
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self._get_feature_names()
        self._is_fitted = True
        return X_processed

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible API: fit on a full frame and return (X, y).

        .. warning::
            This fits on all supplied rows. For leakage-safe training, split
            first and call :meth:`fit` on the training split only. Retained so
            existing callers keep working.
        """
        X = df.drop(columns=[target_column])
        y = np.asarray(df[target_column].values)
        X_processed = self.fit(X, y)
        y_out = self.transform_target(y)
        return X_processed, y_out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform feature data using the fitted preprocessor.

        Accepts frames that may contain extra columns (e.g. a target column on
        inference inputs); only the columns seen during fit are used.
        """
        if not self._is_fitted or self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform")
        features = self._select_input_columns(df)
        return self.preprocessor.transform(features)

    def transform_target(self, y: np.ndarray | pd.Series) -> np.ndarray:
        """Encode target labels if a target encoder was fit, else passthrough."""
        y_arr = np.asarray(y)
        if self.target_encoder is not None:
            return self.target_encoder.transform(y_arr)
        return y_arr

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded target back to original labels."""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y)
        return y

    def _select_input_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the frame restricted/ordered to the columns seen at fit time."""
        if not self.input_columns:
            return df
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing columns required by the preprocessor: {missing}")
        return df[self.input_columns]

    def _get_feature_names(self) -> list[str]:
        """Get processed feature names from the fitted transformer."""
        if self.preprocessor is None:
            return []
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            names: list[str] = []
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
        """Heuristic: decide whether the target is categorical."""
        if y.dtype == object or str(y.dtype).startswith("str"):
            return True
        valid = y[~pd.isna(y)]
        unique = np.unique(valid)
        if len(unique) == len(valid):
            return False
        return len(unique) <= 10 or len(unique) < len(y) * 0.1
