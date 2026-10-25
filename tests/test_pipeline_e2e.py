"""End-to-end pipeline tests (small, fast, deterministic)."""

from pathlib import Path

import numpy as np
import pandas as pd

from lazyml.core.pipeline import LazyMLPipeline


def _make_classification_csv(path: Path, n: int = 200) -> None:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((x1 + x2 + rng.normal(scale=0.3, size=n)) > 0).astype(int)
    df = pd.DataFrame(
        {
            "num1": x1,
            "num2": x2,
            "cat": rng.choice(["a", "b", "c"], size=n),
            "constant": 1.0,
            "target": y,
        }
    )
    df.to_csv(path, index=False)


def test_pipeline_trains_and_predicts(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _make_classification_csv(train_csv)
    out_dir = tmp_path / "out"

    pipeline = LazyMLPipeline(
        target_column="target",
        task_type="classification",
        time_budget=60,
        cv_folds=3,
        output_dir=str(out_dir),
        parallel=False,
        handle_imbalance=False,
    )
    result = pipeline.run(str(train_csv))

    assert result["leaderboard"], "leaderboard should not be empty"
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "preprocessor.pkl").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "leaderboard.json").exists()

    # Accuracy should be meaningfully above chance on this separable data.
    best = pipeline.leaderboard[0]
    assert best.get("accuracy", 0.0) > 0.7


def test_predict_roundtrip_returns_original_labels(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _make_classification_csv(train_csv)
    out_dir = tmp_path / "out"

    pipeline = LazyMLPipeline(
        target_column="target",
        task_type="classification",
        time_budget=60,
        cv_folds=3,
        output_dir=str(out_dir),
        parallel=False,
        handle_imbalance=False,
    )
    pipeline.run(str(train_csv))

    # New data lacking the target column, still containing all feature columns.
    new_csv = tmp_path / "new.csv"
    full = pd.read_csv(train_csv).drop(columns=["target"]).head(15)
    full.to_csv(new_csv, index=False)

    preds = pipeline.predict(str(out_dir / "model.pkl"), str(new_csv))
    assert len(preds) == 15
    # Original labels were 0/1; predictions must be in that label space.
    assert set(np.unique(preds)) <= {0, 1}
