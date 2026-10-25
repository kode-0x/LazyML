"""Main LazyML pipeline (leakage-safe).

Ordering guarantee: the dataset is split into train/test **before** any
stateful fitting. Feature selection, preprocessing (impute/scale/encode) and
class-imbalance resampling are all fit on the training split only, then applied
unchanged to the held-out test split. This prevents test-set information from
leaking into model selection.

The saved inference artifact bundles everything required to reproduce the exact
transformation chain at prediction time: the fitted feature selector, the
fitted preprocessor, and the ordered list of raw input columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lazyml.core.feature_selection import FeatureSelector
from lazyml.core.imbalance_handler import ImbalanceHandler
from lazyml.core.loader import DataLoader
from lazyml.core.preprocessor import DataPreprocessor
from lazyml.core.profiler import DataProfiler
from lazyml.models.selector import ModelSelector
from lazyml.models.trainer import ModelTrainer
from lazyml.reports.generator import ReportGenerator


class InferenceArtifact:
    """Self-contained transformation chain for prediction.

    Bundles the fitted feature selector and preprocessor plus the raw input
    column contract, so :meth:`transform` reproduces training-time preprocessing
    exactly from raw CSV inputs.
    """

    def __init__(
        self,
        raw_feature_columns: list[str],
        feature_selector: FeatureSelector | None,
        preprocessor: DataPreprocessor,
        target_column: str,
        task_type: str,
    ) -> None:
        self.raw_feature_columns = raw_feature_columns
        self.feature_selector = feature_selector
        self.preprocessor = preprocessor
        self.target_column = target_column
        self.task_type = task_type

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw inference data into the model's feature matrix."""
        missing = [c for c in self.raw_feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Prediction input is missing required feature columns: {missing}")
        X = df[self.raw_feature_columns]
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        return self.preprocessor.transform(X)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Map encoded predictions back to original labels."""
        return self.preprocessor.inverse_transform_target(y)


class LazyMLPipeline:
    """End-to-end LazyML pipeline."""

    def __init__(
        self,
        target_column: str | None = None,
        task_type: str | None = None,
        time_budget: int = 300,
        cv_folds: int = 5,
        output_dir: str = "./output",
        ensemble: bool = False,
        feature_selection: bool = True,
        handle_imbalance: bool = True,
        use_optuna: bool = False,
        optuna_trials: int = 30,
        parallel: bool = True,
        max_workers: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.target_column = target_column
        self.task_type = task_type
        self.time_budget = time_budget
        self.cv_folds = cv_folds
        self.output_dir = Path(output_dir)
        self.ensemble = ensemble
        self.feature_selection = feature_selection
        self.handle_imbalance = handle_imbalance
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.parallel = parallel
        self.max_workers = max_workers
        self.random_state = random_state

        self.preprocessor = DataPreprocessor()
        self.feature_selector: FeatureSelector | None = None
        self.imbalance_handler: ImbalanceHandler | None = None
        self.trainer: ModelTrainer | None = None
        self.selector: ModelSelector | None = None
        self.artifact: InferenceArtifact | None = None
        self.best_model: Any = None
        self.leaderboard: list[dict[str, Any]] = []

    def run(self, file_path: str) -> dict[str, Any]:
        """Run the complete leakage-safe LazyML pipeline."""
        # ---- Load & profile -------------------------------------------------
        loader = DataLoader(file_path)
        df = loader.load()

        profiler = DataProfiler(df)
        profile = profiler.profile(self.target_column)

        if self.target_column is None:
            self.target_column = profile.target_column or loader.infer_target_column(df)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        if self.task_type is None:
            self.task_type = profile.task_type or "regression"

        print(f"Target: {self.target_column}, Task: {self.task_type}")
        print(f"Dataset: {profile.n_rows} rows, {profile.n_columns} columns")

        y_full = df[self.target_column]
        X_full = df.drop(columns=[self.target_column])
        raw_feature_columns = list(X_full.columns)

        # ---- Split FIRST (prevents leakage) --------------------------------
        stratify = y_full if self.task_type == "classification" else None
        X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify,
        )

        y_train_raw = np.asarray(y_train_raw)
        y_test_raw = np.asarray(y_test_raw)

        # ---- Feature selection: fit on TRAIN only --------------------------
        if self.feature_selection:
            print("  Running feature selection (train split only)...")
            self.feature_selector = FeatureSelector(
                task_type=self.task_type,
                variance_threshold=0.01,
                correlation_threshold=0.95,
                use_model_selection=True,
                max_features=None,
            )
            X_train_df, selected = self.feature_selector.fit_transform(X_train_df, y_train_raw)
            X_test_df = self.feature_selector.transform(X_test_df)
            print(f"  Selected {len(selected)} features")

        # ---- Preprocessing: fit on TRAIN only ------------------------------
        print("  Preprocessing (fit on train, apply to test)...")
        X_train = self.preprocessor.fit(X_train_df, y_train_raw)
        X_test = self.preprocessor.transform(X_test_df)
        y_train = self.preprocessor.transform_target(y_train_raw)
        y_test = self.preprocessor.transform_target(y_test_raw)

        # ---- Imbalance handling: resample TRAIN only -----------------------
        if self.handle_imbalance and self.task_type == "classification":
            self.imbalance_handler = ImbalanceHandler()
            info = self.imbalance_handler.analyze(y_train)
            if info["is_imbalanced"]:
                print(f"  Imbalance detected: ratio={info['imbalance_ratio']:.2f}")
                X_train, y_train = self.imbalance_handler.resample_array(X_train, y_train)

        # ---- Train --------------------------------------------------------
        self.trainer = ModelTrainer(
            task_type=self.task_type,
            cv_folds=self.cv_folds,
            time_budget=self.time_budget,
            ensemble=self.ensemble,
            parallel=self.parallel,
            max_workers=self.max_workers,
            use_optuna=self.use_optuna,
            optuna_trials=self.optuna_trials,
        )
        results = self.trainer.train(X_train, y_train)

        # ---- Select best on held-out test ---------------------------------
        self.selector = ModelSelector(self.task_type)
        self.best_model, self.leaderboard = self.selector.select_best(results, X_test, y_test)

        # ---- Build inference artifact & save ------------------------------
        self.artifact = InferenceArtifact(
            raw_feature_columns=raw_feature_columns,
            feature_selector=self.feature_selector,
            preprocessor=self.preprocessor,
            target_column=self.target_column,
            task_type=self.task_type,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_outputs(df)

        return {
            "best_model": self.best_model,
            "leaderboard": self.leaderboard,
            "output_dir": str(self.output_dir),
        }

    def predict(self, model_path: str, data_path: str) -> np.ndarray:
        """Load a saved model artifact and make predictions on raw CSV data."""
        model_data = joblib.load(model_path)
        model = model_data["model"]
        artifact = model_data.get("artifact")

        loader = DataLoader(data_path)
        df = loader.load()

        if artifact is not None:
            X = artifact.transform(df)
            preds = model.predict(X)
            return artifact.inverse_transform_target(preds)

        # Legacy fallback: bare preprocessor only.
        preprocessor = model_data["preprocessor"]
        X = preprocessor.transform(df)
        return model.predict(X)

    def _save_outputs(self, df: pd.DataFrame) -> None:
        """Persist model bundle, preprocessor, and reports."""
        joblib.dump(
            {
                "model": self.best_model,
                "artifact": self.artifact,
                # Kept for backward compatibility with older loaders.
                "preprocessor": self.preprocessor,
            },
            self.output_dir / "model.pkl",
        )
        joblib.dump(self.preprocessor, self.output_dir / "preprocessor.pkl")

        assert self.target_column is not None  # set during run()
        assert self.task_type is not None  # set during run()
        report_gen = ReportGenerator(self.output_dir)
        report_gen.generate(
            df=df,
            target_column=self.target_column,
            task_type=self.task_type,
            leaderboard=self.leaderboard,
            best_model_name=self.leaderboard[0]["model"] if self.leaderboard else "Unknown",
        )

        print(f"\nOutputs saved to: {self.output_dir}")
