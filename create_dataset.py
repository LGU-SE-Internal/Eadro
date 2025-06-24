#!/usr/bin/env python3
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich import print as rprint
from src.eadro.data_adapter import create_dataset

app = typer.Typer(
    help="Create and manage datasets with incremental processing for EADRO"
)
console = Console()


@app.command()
def main(
    data_root: str = typer.Argument(
        "/mnt/jfs/rcabench_dataset",
        help="Root directory containing data packs",
    ),
    output_dir: str = typer.Argument(
        "dataset_output", help="Output directory for processed dataset"
    ),
    max_cases: Optional[int] = typer.Option(
        None, "--max-cases", help="Maximum number of cases to process"
    ),
    chunk_length: int = typer.Option(10, "--chunk-length", help="Length of each chunk"),
    train_ratio: float = typer.Option(
        0.9, "--train-ratio", help="Ratio of training data", min=0.0, max=1.0
    ),
) -> None:
    if not Path(data_root).exists():
        rprint(f"[red]Error: Data root directory '{data_root}' does not exist[/red]")
        raise typer.Exit(1)

    try:
        total_chunks = create_dataset(
            data_root=data_root,
            output_dir=output_dir,
            max_cases=max_cases,
            chunk_length=chunk_length,
            train_ratio=train_ratio,
        )

        console.print(f"[green]Total chunks created: {total_chunks}[/green]")
        console.print(f"[green]Dataset saved to: {output_dir}[/green]")

    except KeyboardInterrupt:
        console.print("[yellow]Dataset creation interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        import traceback

        console.print(f"[red]Error: {e}[/red]")
        console.print("[red]Full traceback:[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
