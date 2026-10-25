"""Property-based tests using Hypothesis."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from lazyml.core.preprocessor import DataPreprocessor


@settings(max_examples=30, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    )
)
def test_preprocessor_output_row_count_matches_input(values: list[float]) -> None:
    """Transform must preserve the number of rows for any numeric input."""
    X = pd.DataFrame({"x": values})
    y = np.array([i % 2 for i in range(len(values))])
    pre = DataPreprocessor()
    pre.fit(X, y)
    out = pre.transform(X)
    assert out.shape[0] == len(values)


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=4, max_value=40),
)
def test_target_label_encoding_is_bijective(n: int) -> None:
    """Encoding then inverse-encoding a categorical target returns the labels."""
    labels = np.array(["yes" if i % 2 else "no" for i in range(n)])
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    pre = DataPreprocessor()
    pre.fit(X, labels)
    encoded = pre.transform_target(labels)
    decoded = pre.inverse_transform_target(encoded)
    assert list(decoded) == list(labels)
