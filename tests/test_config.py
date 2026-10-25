"""Tests for the Pydantic configuration layer."""

from pathlib import Path

import pytest

from lazyml.config import LazyMLConfig


def test_defaults_match_cli_historical_values() -> None:
    cfg = LazyMLConfig()
    assert cfg.target is None
    assert cfg.task is None
    assert cfg.output_dir == Path("./output")
    assert cfg.feature_selection is True
    assert cfg.handle_imbalance is True
    assert cfg.parallel is True
    assert cfg.ensemble is False
    assert cfg.random_state == 42
    assert cfg.optimization.time_budget == 300
    assert cfg.optimization.n_trials == 30
    assert cfg.cross_validation.folds == 5


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_text = """
    target: price
    task: regression
    optimization:
      time_budget: 600
      n_trials: 50
      use_optuna: true
    cross_validation:
      folds: 3
      strategy: kfold
    models:
      include: [random_forest]
      exclude: [svm]
    """
    p = tmp_path / "lazyml.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    cfg = LazyMLConfig.from_yaml(p)
    assert cfg.target == "price"
    assert cfg.task == "regression"
    assert cfg.optimization.time_budget == 600
    assert cfg.optimization.use_optuna is True
    assert cfg.cross_validation.folds == 3
    assert cfg.models.include == ["random_forest"]


def test_task_auto_normalizes_to_none() -> None:
    assert LazyMLConfig(task="auto").task is None
    assert LazyMLConfig(task="CLASSIFICATION").task == "classification"


def test_invalid_task_rejected() -> None:
    with pytest.raises(ValueError):
        LazyMLConfig(task="clustering")


def test_include_exclude_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        LazyMLConfig(models={"include": ["rf"], "exclude": ["rf"]})


def test_extra_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "lazyml.yaml"
    p.write_text("not_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LazyMLConfig.from_yaml(p)


def test_discover_returns_none_when_absent(tmp_path: Path) -> None:
    assert LazyMLConfig.discover(tmp_path) is None


def test_discover_loads_present_file(tmp_path: Path) -> None:
    (tmp_path / "lazyml.yaml").write_text("target: y\n", encoding="utf-8")
    cfg = LazyMLConfig.discover(tmp_path)
    assert cfg is not None
    assert cfg.target == "y"


def test_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LazyMLConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "lazyml.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LazyMLConfig.from_yaml(p)
