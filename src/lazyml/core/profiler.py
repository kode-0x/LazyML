"""Dataset profiling and analysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype


@dataclass
class ColumnProfile:
    """Profile for a single column."""

    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    is_text: bool


@dataclass
class DatasetProfile:
    """Complete dataset profile."""

    n_rows: int
    n_columns: int
    memory_usage_mb: float
    columns: list[ColumnProfile]
    has_missing_values: bool
    duplicate_rows: int
    target_column: str | None = None
    task_type: str | None = None  # 'classification' or 'regression'


class DataProfiler:
    """Profile datasets for analysis."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def profile(self, target_column: str | None = None) -> DatasetProfile:
        """Generate comprehensive dataset profile."""
        n_rows, n_columns = self.df.shape
        memory_usage_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)

        columns = []
        for col_name in self.df.columns:
            col = self.df[col_name]
            profile = self._profile_column(col_name, col)
            columns.append(profile)

        has_missing = any(col.missing_count > 0 for col in columns)
        duplicate_rows = self.df.duplicated().sum()

        # Infer target and task type if not provided
        if target_column is None:
            target_column = self._infer_target_column()

        task_type = None
        if target_column and target_column in self.df.columns:
            task_type = self._detect_task_type(target_column)

        return DatasetProfile(
            n_rows=n_rows,
            n_columns=n_columns,
            memory_usage_mb=memory_usage_mb,
            columns=columns,
            has_missing_values=has_missing,
            duplicate_rows=duplicate_rows,
            target_column=target_column,
            task_type=task_type,
        )

    def _profile_column(self, name: str, series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        missing_count = series.isna().sum()
        missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0
        unique_count = series.nunique(dropna=True)

        is_numeric = is_numeric_dtype(series)
        is_categorical = is_categorical_dtype(series) or (
            is_object_dtype(series) and unique_count <= min(50, len(series) * 0.05)
        )
        is_text = is_object_dtype(series) and not is_categorical

        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            missing_count=int(missing_count),
            missing_pct=round(missing_pct, 2),
            unique_count=int(unique_count),
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_text=is_text,
        )

    def _infer_target_column(self) -> str | None:
        """Infer target column from common patterns."""
        common_names = ["target", "label", "y", "class", "output"]
        for col in self.df.columns:
            if col.lower() in common_names:
                return col
        return None

    def _detect_task_type(self, target_column: str) -> str:
        """Detect if task is classification or regression."""
        target = self.df[target_column]

        if is_numeric_dtype(target):
            unique_count = target.nunique(dropna=True)
            if unique_count <= 10:
                return "classification"
            return "regression"

        return "classification"
