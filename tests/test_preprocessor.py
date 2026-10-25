"""Tests asserting leakage-safe preprocessing behavior."""

import numpy as np
import pandas as pd
import pytest

from lazyml.core.preprocessor import DataPreprocessor


def test_scaler_uses_train_statistics_only() -> None:
    """Test data must be scaled with the TRAIN mean/std, not its own.

    A constant test feature at the train mean should map to ~0 after scaling.
    """
    X_train = pd.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0]})
    y_train = np.array([0, 1, 0, 1, 0])
    pre = DataPreprocessor()
    pre.fit(X_train, y_train)

    train_mean = X_train["x"].mean()  # 20.0
    X_test = pd.DataFrame({"x": [train_mean, train_mean]})
    transformed = pre.transform(X_test)
    # If scaled with train stats, values at the train mean become ~0.
    assert np.allclose(transformed, 0.0, atol=1e-9)


def test_transform_does_not_refit() -> None:
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0])
    pre = DataPreprocessor()
    pre.fit(X_train, y)
    scaler = pre.preprocessor.named_transformers_["num"].named_steps["scaler"]
    mean_before = scaler.mean_.copy()

    pre.transform(pd.DataFrame({"x": [100.0, 200.0]}))
    scaler_after = pre.preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert np.array_equal(mean_before, scaler_after.mean_)


def test_target_encoding_roundtrip() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = np.array(["cat", "dog", "cat", "dog"])
    pre = DataPreprocessor()
    pre.fit(X, y)
    encoded = pre.transform_target(y)
    assert set(np.unique(encoded)) <= {0, 1}
    decoded = pre.inverse_transform_target(encoded)
    assert list(decoded) == list(y)


def test_transform_before_fit_raises() -> None:
    pre = DataPreprocessor()
    with pytest.raises(ValueError):
        pre.transform(pd.DataFrame({"x": [1.0]}))


def test_transform_missing_column_raises() -> None:
    pre = DataPreprocessor()
    pre.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), np.array([0, 1]))
    with pytest.raises(ValueError):
        pre.transform(pd.DataFrame({"a": [1.0]}))


def test_unknown_category_handled_at_transform() -> None:
    X_train = pd.DataFrame({"c": ["a", "b", "a", "b"]})
    pre = DataPreprocessor()
    pre.fit(X_train, np.array([0, 1, 0, 1]))
    # Unseen category "z" must not raise.
    out = pre.transform(pd.DataFrame({"c": ["z"]}))
    assert out.shape[0] == 1
