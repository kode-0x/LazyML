"""Tests for dataset profiling, including pandas string-dtype handling."""

import pandas as pd

from lazyml.core.profiler import DataProfiler


def _col(profile, name):
    return next(c for c in profile.columns if c.name == name)


def test_numeric_column_detected() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    profile = DataProfiler(df).profile()
    x = _col(profile, "x")
    assert x.is_numeric
    assert not x.is_categorical
    assert not x.is_text


def test_low_cardinality_string_is_categorical() -> None:
    # Works whether pandas stores this as object (2.x) or str (3.x).
    df = pd.DataFrame({"c": ["a", "b", "a", "b"] * 10, "target": [0, 1] * 20})
    profile = DataProfiler(df).profile()
    c = _col(profile, "c")
    assert not c.is_numeric
    assert c.is_categorical
    assert not c.is_text


def test_high_cardinality_string_is_text() -> None:
    df = pd.DataFrame({"c": [f"id_{i}" for i in range(200)], "target": [0, 1] * 100})
    profile = DataProfiler(df).profile()
    c = _col(profile, "c")
    assert not c.is_numeric
    assert c.is_text
    assert not c.is_categorical


def test_missing_values_and_duplicates() -> None:
    df = pd.DataFrame({"x": [1.0, None, 1.0], "target": [0, 1, 0]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    profile = DataProfiler(df).profile()
    assert profile.has_missing_values
    assert profile.duplicate_rows >= 1


def test_target_and_task_inference() -> None:
    df = pd.DataFrame({"f": range(100), "target": [i % 3 for i in range(100)]})
    profile = DataProfiler(df).profile()
    assert profile.target_column == "target"
    assert profile.task_type == "classification"

    df_reg = pd.DataFrame({"f": range(100), "target": [float(i) for i in range(100)]})
    profile_reg = DataProfiler(df_reg).profile()
    assert profile_reg.task_type == "regression"
