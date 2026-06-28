"""Main LazyML pipeline."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lazyml.core.loader import DataLoader
from lazyml.core.preprocessor import DataPreprocessor
from lazyml.core.profiler import DataProfiler
from lazyml.core.feature_selection import FeatureSelector
from lazyml.core.imbalance_handler import ImbalanceHandler
from lazyml.models.trainer import ModelTrainer
from lazyml.models.selector import ModelSelector
from lazyml.reports.generator import ReportGenerator


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

        self.preprocessor = DataPreprocessor()
        self.feature_selector: FeatureSelector | None = None
        self.imbalance_handler: ImbalanceHandler | None = None
        self.trainer: ModelTrainer | None = None
        self.selector: ModelSelector | None = None
        self.best_model: Any = None
        self.leaderboard: list[dict[str, Any]] = []

    def run(self, file_path: str) -> dict[str, Any]:
        """Run complete LazyML pipeline."""
        # Load data
        loader = DataLoader(file_path)
        df = loader.load()

        # Profile data
        profiler = DataProfiler(df)
        profile = profiler.profile(self.target_column)

        # Auto-detect target and task if needed
        if self.target_column is None:
            self.target_column = profile.target_column or loader.infer_target_column(df)

        if self.task_type is None:
            self.task_type = profile.task_type or "regression"

        print(f"Target: {self.target_column}, Task: {self.task_type}")
        print(f"Dataset: {profile.n_rows} rows, {profile.n_columns} columns")

        # Separate target
        y = df[self.target_column].values
        X_df = df.drop(columns=[self.target_column])

        # Handle class imbalance for classification
        if self.handle_imbalance and self.task_type == "classification":
            self.imbalance_handler = ImbalanceHandler()
            imbalance_info = self.imbalance_handler.analyze(y)
            if imbalance_info["is_imbalanced"]:
                print(f"  Imbalance detected: ratio={imbalance_info['imbalance_ratio']:.2f}")
                X_df, y = self.imbalance_handler.resample(X_df, y)

        # Feature selection
        if self.feature_selection:
            print("  Running feature selection...")
            self.feature_selector = FeatureSelector(
                task_type=self.task_type,
                variance_threshold=0.01,
                correlation_threshold=0.95,
                use_model_selection=True,
                max_features=None,
            )
            X_df, selected_features = self.feature_selector.fit_transform(X_df, y)
            print(f"  Selected {len(selected_features)} features")

        # Preprocess
        print("  Preprocessing...")
        # Temporarily recombine for preprocessing
        df_processed = X_df.copy()
        df_processed[self.target_column] = y
        X, y = self.preprocessor.fit_transform(df_processed, self.target_column)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == "classification" else None
        )

        # Train models
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

        # Select best model
        self.selector = ModelSelector(self.task_type)
        self.best_model, self.leaderboard = self.selector.select_best(results, X_test, y_test)

        # Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_outputs(df, X_test, y_test)

        return {
            "best_model": self.best_model,
            "leaderboard": self.leaderboard,
            "output_dir": str(self.output_dir),
        }

    def predict(self, model_path: str, data_path: str) -> np.ndarray:
        """Load model and make predictions."""
        # Load model and preprocessor
        model_data = joblib.load(model_path)
        model = model_data["model"]
        preprocessor = model_data["preprocessor"]

        # Load and preprocess data
        loader = DataLoader(data_path)
        df = loader.load()
        X = preprocessor.transform(df)

        return model.predict(X)

    def _save_outputs(self, df: pd.DataFrame, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Save all outputs."""
        # Save model
        joblib.dump(
            {"model": self.best_model, "preprocessor": self.preprocessor},
            self.output_dir / "model.pkl"
        )

        # Save preprocessor separately
        joblib.dump(self.preprocessor, self.output_dir / "preprocessor.pkl")

        # Generate report
        report_gen = ReportGenerator(self.output_dir)
        report_gen.generate(
            df=df,
            target_column=self.target_column,
            task_type=self.task_type,
            leaderboard=self.leaderboard,
            best_model_name=self.leaderboard[0]["model"] if self.leaderboard else "Unknown",
        )

        print(f"\nOutputs saved to: {self.output_dir}")
