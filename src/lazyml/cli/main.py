"""Main CLI entry point using Typer."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lazyml.core.loader import DataLoader
from lazyml.core.pipeline import LazyMLPipeline
from lazyml.core.profiler import DataProfiler

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
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    """LazyML - LazyML for CSV datasets."""
    pass


@app.command()
def run(
    file: Optional[Path] = typer.Argument(
        None,
        help="Path to CSV dataset (optional - runs interactive mode if not provided)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name (auto-detect if not specified)"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Task type: classification or regression (auto-detect if not specified)"
    ),
    time_budget: int = typer.Option(
        300, "--time-budget", help="Maximum training time in seconds"
    ),
    cv_folds: int = typer.Option(
        5, "--cv-folds", help="Number of cross-validation folds"
    ),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", "-o", help="Directory for output files"
    ),
    ensemble: bool = typer.Option(
        False, "--ensemble", help="Enable ensemble methods"
    ),
    feature_selection: bool = typer.Option(
        True, "--feature-selection/--no-feature-selection", help="Enable automatic feature selection"
    ),
    handle_imbalance: bool = typer.Option(
        True, "--handle-imbalance/--no-handle-imbalance", help="Handle class imbalance for classification"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", help="Train models in parallel"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum parallel workers (default: auto)"
    ),
    use_optuna: bool = typer.Option(
        False, "--optuna", help="Enable Optuna hyperparameter tuning"
    ),
    optuna_trials: int = typer.Option(
        30, "--optuna-trials", help="Number of Optuna trials per model"
    ),
) -> None:
    """Train models on a dataset."""
    print(Panel.fit("[bold green]LazyML[/bold green]", border_style="green"))

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

            target_input = typer.prompt(f"Target column", default=suggested_target)
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
        )

        result = pipeline.run(str(file))

        # Success output
        print("\n[bold green]Training Complete![/bold green]")
        print(f"Best model: [cyan]{result['leaderboard'][0]['model']}[/cyan]")
        print(f"Output directory: [cyan]{result['output_dir']}[/cyan]")

    except Exception as e:
        print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model: Path = typer.Argument(..., help="Path to trained model pickle file", exists=True),
    data: Path = typer.Argument(..., help="Path to CSV file for prediction", exists=True),
    output: Path = typer.Option(
        Path("predictions.csv"), "--output", "-o", help="Output file path"
    ),
) -> None:
    """Generate predictions using a trained model."""
    print(Panel.fit("[bold blue]LazyML Predict[/bold blue]", border_style="blue"))

    import joblib
    import pandas as pd

    try:
        # Load model
        model_data = joblib.load(model)
        loaded_model = model_data["model"]
        preprocessor = model_data["preprocessor"]

        # Load and preprocess data
        loader = DataLoader(data)
        df = loader.load()
        X = preprocessor.transform(df)

        # Predict
        predictions = loaded_model.predict(X)

        # Inverse transform if classification
        if preprocessor.target_encoder is not None:
            predictions = preprocessor.target_encoder.inverse_transform(predictions)

        # Save
        output_df = pd.DataFrame({"prediction": predictions})
        output_df.to_csv(output, index=False)

        print(f"[green]Predictions saved to: {output}[/green]")
        print(f"Generated {len(predictions)} predictions")

    except Exception as e:
        print(f"[red]Error during prediction: {e}[/red]")
        raise typer.Exit(1)


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
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
