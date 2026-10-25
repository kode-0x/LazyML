"""Typed configuration for LazyML using Pydantic v2.

Configuration can be supplied via a ``lazyml.yaml`` file and/or CLI flags.
CLI flags always take precedence over file values. All fields have defaults
that preserve the historical CLI behavior, so configuration is fully optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["classification", "regression"]


class ModelsConfig(BaseModel):
    """Which models to include or exclude from training."""

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class OptimizationConfig(BaseModel):
    """Hyperparameter optimization settings."""

    model_config = ConfigDict(extra="forbid")

    time_budget: int = Field(default=300, gt=0)
    use_optuna: bool = False
    n_trials: int = Field(default=30, gt=0)


class CrossValidationConfig(BaseModel):
    """Cross-validation settings."""

    model_config = ConfigDict(extra="forbid")

    folds: int = Field(default=5, ge=2)
    strategy: Literal["auto", "stratified", "kfold"] = "auto"


class LazyMLConfig(BaseModel):
    """Top-level LazyML configuration.

    Mirrors the ``run`` command options so a ``lazyml.yaml`` file can set
    persistent defaults. Extra keys are rejected to catch typos early.
    """

    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    task: TaskType | None = None
    output_dir: Path = Path("./output")
    ensemble: bool = False
    feature_selection: bool = True
    handle_imbalance: bool = True
    parallel: bool = True
    max_workers: int | None = Field(default=None, ge=1)
    random_state: int = 42

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)

    @field_validator("task", mode="before")
    @classmethod
    def _normalize_task(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("", "auto", "none"):
                return None
            return lowered
        return value

    @model_validator(mode="after")
    def _check_model_lists(self) -> LazyMLConfig:
        overlap = set(self.models.include) & set(self.models.exclude)
        if overlap:
            raise ValueError(f"Models cannot be both included and excluded: {sorted(overlap)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> LazyMLConfig:
        """Load configuration from a YAML file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file is not a valid YAML mapping or fails
                schema validation.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML config {p}: {e}") from e
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file {p} must contain a mapping at the top level, got {type(raw).__name__}"
            )
        return cls.model_validate(raw)

    @classmethod
    def discover(cls, start: str | Path | None = None) -> LazyMLConfig | None:
        """Find and load ``lazyml.yaml`` from the given directory, if present.

        Returns ``None`` when no config file exists so callers can fall back to
        pure defaults without special-casing.
        """
        base = Path(start) if start is not None else Path.cwd()
        for name in ("lazyml.yaml", "lazyml.yml"):
            candidate = base / name
            if candidate.exists():
                return cls.from_yaml(candidate)
        return None
