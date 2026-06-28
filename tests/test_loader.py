"""Tests for data loader."""

import pytest
import pandas as pd
from pathlib import Path

from lazyml.core.loader import DataLoader


def test_load_csv(tmp_path: Path) -> None:
    """Test loading a valid CSV file."""
    # Create test CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
        "target": [0, 1, 0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(csv_path)
    loaded = loader.load()

    assert len(loaded) == 3
    assert list(loaded.columns) == ["feature1", "feature2", "target"]


def test_load_nonexistent_file() -> None:
    """Test loading a non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        loader = DataLoader("nonexistent.csv")
        loader.load()


def test_load_non_csv() -> None:
    """Test loading non-CSV file raises error."""
    with pytest.raises(ValueError):
        loader = DataLoader("file.txt")
        loader.load()


def test_infer_target_column() -> None:
    """Test target column inference."""
    df = pd.DataFrame({
        "a": [1, 2],
        "target": [0, 1],
    })
    target = DataLoader.infer_target_column(df)
    assert target == "target"
