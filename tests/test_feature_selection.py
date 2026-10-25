"""Tests for feature selection fit/transform consistency."""

import numpy as np
import pandas as pd
import pytest

from lazyml.core.feature_selection import FeatureSelector


def test_transform_matches_selected_columns() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "informative": rng.normal(size=100),
            "constant": np.ones(100),  # dropped by variance threshold
            "noise": rng.normal(size=100),
        }
    )
    y = (X["informative"] > 0).astype(int).to_numpy()

    fs = FeatureSelector(task_type="classification", variance_threshold=0.01)
    X_train_sel, selected = fs.fit_transform(X, y)
    assert "constant" not in selected

    X_new = X.iloc[:10].copy()
    X_new_sel = fs.transform(X_new)
    assert list(X_new_sel.columns) == selected


def test_transform_before_fit_raises() -> None:
    fs = FeatureSelector(task_type="classification")
    with pytest.raises(ValueError):
        fs.transform(pd.DataFrame({"x": [1.0]}))


def test_transform_missing_column_raises() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0])
    fs = FeatureSelector(task_type="classification", variance_threshold=0.0)
    _, selected = fs.fit_transform(X, y)
    with pytest.raises(ValueError):
        fs.transform(pd.DataFrame({"a": [1.0]}))
