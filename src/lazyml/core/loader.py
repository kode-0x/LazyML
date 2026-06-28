"""Data loading utilities."""

from pathlib import Path
from typing import Union

import pandas as pd


class DataLoader:
    """Load CSV datasets with validation."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """Load and validate CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected CSV file, got: {self.file_path.suffix}")

        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}") from e

        if df.empty:
            raise ValueError("Dataset is empty")

        if len(df.columns) < 2:
            raise ValueError("Dataset must have at least 2 columns (features + target)")

        return df

    @staticmethod
    def infer_target_column(df: pd.DataFrame) -> str:
        """Infer target column from common names."""
        common_targets = [
            "target",
            "label",
            "y",
            "class",
            "output",
            "prediction",
            "result",
        ]

        for col in df.columns:
            lower_col = col.lower()
            if lower_col in common_targets:
                return col

        # Default to last column if no common target found
        return df.columns[-1]
