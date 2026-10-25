"""Main CLI entry point using Typer."""

from pathlib import Path
from typing import Any

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lazyml.config import LazyMLConfig
from lazyml.core.loader import DataLoader
from lazyml.core.pipeline import LazyMLPipeline
from lazyml.core.profiler import DataProfiler


def _explicit_params(ctx: typer.Context | None) -> set[str]:
    """Return the set of parameter names the user set explicitly on the CLI.

    Used so a loaded config only fills in values the user did not pass. If the
    click context is unavailable, an empty set is returned (config wins over
    historical defaults, which is still backward compatible because defaults
    and config defaults match).
    """
    if ctx is None:
        return set()
    explicit: set[str] = set()
    for name in ctx.params:
        source = ctx.get_parameter_source(name) if hasattr(ctx, "get_parameter_source") else None
        if source is not None and source.name == "COMMANDLINE":
            explicit.add(name)
    return explicit


app = typer.Typer(
    name="lazyml",
    help="LazyML command-line tool for automated machine learning on CSV datasets.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        print("LazyML 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True),
) -> None:
    """LazyML - LazyML for CSV datasets."""
    pass


@app.command()
def run(
    file: Path | None = typer.Argument(
        None,
        help="Path to CSV dataset (optional - runs interactive mode if not provided)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target column name (auto-detect if not specified)"
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Task type: classification or regression (auto-detect if not specified)",
    ),
    time_budget: int = typer.Option(300, "--time-budget", help="Maximum training time in seconds"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds"),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", "-o", help="Directory for output files"
    ),
    ensemble: bool = typer.Option(False, "--ensemble", help="Enable ensemble methods"),
    feature_selection: bool = typer.Option(
        True,
        "--feature-selection/--no-feature-selection",
        help="Enable automatic feature selection",
    ),
    handle_imbalance: bool = typer.Option(
        True,
        "--handle-imbalance/--no-handle-imbalance",
        help="Handle class imbalance for classification",
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", help="Train models in parallel"
    ),
    max_workers: int | None = typer.Option(
        None, "--max-workers", help="Maximum parallel workers (default: auto)"
    ),
    use_optuna: bool = typer.Option(False, "--optuna", help="Enable Optuna hyperparameter tuning"),
    optuna_trials: int = typer.Option(
        30, "--optuna-trials", help="Number of Optuna trials per model"
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to lazyml.yaml config (auto-discovered in CWD if omitted)",
    ),
    ctx: typer.Context = None,  # type: ignore[assignment]  # Typer injects the context
) -> None:
    """Train models on a dataset."""
    print(Panel.fit("[bold green]LazyML[/bold green]", border_style="green"))

    # Load optional configuration. Explicit CLI flags always win; config only
    # fills in values the user did not set on the command line.
    cfg = None
    try:
        cfg = LazyMLConfig.from_yaml(config) if config is not None else LazyMLConfig.discover()
    except (FileNotFoundError, ValueError) as e:
        print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(1) from e

    if cfg is not None:
        explicit = _explicit_params(ctx)

        def pick(name: str, cli_value: Any, cfg_value: Any) -> Any:
            return cli_value if name in explicit else cfg_value

        target = pick("target", target, cfg.target)
        task = pick("task", task, cfg.task)
        output_dir = pick("output_dir", output_dir, cfg.output_dir)
        ensemble = pick("ensemble", ensemble, cfg.ensemble)
        feature_selection = pick("feature_selection", feature_selection, cfg.feature_selection)
        handle_imbalance = pick("handle_imbalance", handle_imbalance, cfg.handle_imbalance)
        parallel = pick("parallel", parallel, cfg.parallel)
        max_workers = pick("max_workers", max_workers, cfg.max_workers)
        time_budget = pick("time_budget", time_budget, cfg.optimization.time_budget)
        cv_folds = pick("cv_folds", cv_folds, cfg.cross_validation.folds)
        use_optuna = pick("use_optuna", use_optuna, cfg.optimization.use_optuna)
        optuna_trials = pick("optuna_trials", optuna_trials, cfg.optimization.n_trials)

    # Interactive mode if no file provided
    if file is None:
        print("[yellow]Interactive mode - please provide the following:[/yellow]")
        file_path = typer.prompt("Path to CSV file")
        file = Path(file_path)
        if not file.exists():
            print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(1)

        # Profile to suggest target
        try:
            loader = DataLoader(file)
            df = loader.load()
            profiler = DataProfiler(df)
            profile = profiler.profile()
            suggested_target = profile.target_column or df.columns[-1]

            target_input = typer.prompt("Target column", default=suggested_target)
            target = target_input if target_input else None
        except Exception as e:
            print(f"[red]Error profiling dataset: {e}[/red]")
            target = typer.prompt("Target column")

        if task is None:
            task_input = typer.prompt("Task type (classification/regression)", default="auto")
            task = None if task_input == "auto" else task_input

    # Validate file
    if not file.exists():
        print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)

    # Run pipeline
    try:
        pipeline = LazyMLPipeline(
            target_column=target,
            task_type=task,
            time_budget=time_budget,
            cv_folds=cv_folds,
            output_dir=str(output_dir),
            ensemble=ensemble,
            feature_selection=feature_selection,
            handle_imbalance=handle_imbalance,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            max_workers=max_workers,
            parallel=parallel,
            random_state=cfg.random_state if cfg is not None else 42,
        )

        result = pipeline.run(str(file))

        # Success output
        print("\n[bold green]Training Complete![/bold green]")
        print(f"Best model: [cyan]{result['leaderboard'][0]['model']}[/cyan]")
        print(f"Output directory: [cyan]{result['output_dir']}[/cyan]")

    except Exception as e:
        print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def predict(
    model: Path = typer.Argument(..., help="Path to trained model pickle file", exists=True),
    data: Path = typer.Argument(..., help="Path to CSV file for prediction", exists=True),
    output: Path = typer.Option(Path("predictions.csv"), "--output", "-o", help="Output file path"),
) -> None:
    """Generate predictions using a trained model."""
    print(Panel.fit("[bold blue]LazyML Predict[/bold blue]", border_style="blue"))

    import joblib
    import pandas as pd

    try:
        model_data = joblib.load(model)
        loaded_model = model_data["model"]
        artifact = model_data.get("artifact")

        loader = DataLoader(data)
        df = loader.load()

        if artifact is not None:
            # Preferred path: full leakage-safe transformation chain.
            X = artifact.transform(df)
            predictions = loaded_model.predict(X)
            predictions = artifact.inverse_transform_target(predictions)
        else:
            # Legacy fallback for older model bundles.
            preprocessor = model_data["preprocessor"]
            X = preprocessor.transform(df)
            predictions = loaded_model.predict(X)
            if preprocessor.target_encoder is not None:
                predictions = preprocessor.target_encoder.inverse_transform(predictions)

        output_df = pd.DataFrame({"prediction": predictions})
        output_df.to_csv(output, index=False)

        print(f"[green]Predictions saved to: {output}[/green]")
        print(f"Generated {len(predictions)} predictions")

    except Exception as e:
        print(f"[red]Error during prediction: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def profile(
    file: Path = typer.Argument(..., help="Path to CSV file", exists=True),
) -> None:
    """Analyze dataset without training."""
    print(Panel.fit("[bold yellow]LazyML Dataset Profile[/bold yellow]", border_style="yellow"))

    try:
        loader = DataLoader(file)
        df = loader.load()

        profiler = DataProfiler(df)
        profile = profiler.profile()

        # Display summary
        table = Table(title="Dataset Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Rows", f"{profile.n_rows:,}")
        table.add_row("Columns", str(profile.n_columns))
        table.add_row("Memory", f"{profile.memory_usage_mb:.2f} MB")
        table.add_row("Missing Values", "Yes" if profile.has_missing_values else "No")
        table.add_row("Duplicate Rows", str(profile.duplicate_rows))
        table.add_row("Suggested Target", profile.target_column or "Unknown")
        table.add_row("Detected Task", profile.task_type or "Unknown")

        console.print(table)

        # Column details
        col_table = Table(title="Column Details")
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Type", style="blue")
        col_table.add_column("Missing", style="yellow")
        col_table.add_column("Unique", style="green")

        for col in profile.columns:
            col_table.add_row(
                col.name,
                col.dtype,
                f"{col.missing_count} ({col.missing_pct}%)",
                str(col.unique_count),
            )

        console.print(col_table)

    except Exception as e:
        print(f"[red]Error profiling dataset: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
