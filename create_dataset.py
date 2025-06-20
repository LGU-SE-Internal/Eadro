from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.eadro.data_adapter import create_dataset_streaming

app = typer.Typer(
    help="EADRO Dataset Creation Tool - Create complete datasets from multiple cases"
)
console = Console()


@app.command()
def main(
    data_root: str = typer.Option(
        "/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/",
        "--data-root",
        help="Root directory containing case data",
    ),
    cases_file: str = typer.Option(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet",
        "--cases-file",
        help="Parquet file containing case index",
    ),
    output_dir: str = typer.Option(
        "dataset_output",
        "--output-dir",
        help="Output directory for the complete dataset",
    ),
    max_cases: Optional[int] = typer.Option(
        10, "--max-cases", help="Maximum number of cases to process (None for all)"
    ),
    batch_size: int = typer.Option(
        2, "--batch-size", help="Number of cases to process in each batch"
    ),
    chunk_length: int = typer.Option(
        10, "--chunk-length", help="Length of data chunks"
    ),
    train_ratio: float = typer.Option(
        0.7, "--train-ratio", help="Ratio of training data (0.0-1.0)"
    ),
):
    console.print("🚀 [bold blue]Starting EADRO Dataset Creation[/bold blue]")
    console.print()

    console.print("[bold green]Configuration:[/bold green]")
    console.print(f"  📁 Data root: {data_root}")
    console.print(f"  📊 Cases file: {cases_file}")
    console.print(f"  💾 Output dir: {output_dir}")
    console.print(f"  🔢 Max cases: {max_cases if max_cases else 'All'}")
    console.print(f"  📦 Batch size: {batch_size}")
    console.print(f"  📏 Chunk length: {chunk_length}")
    console.print(f"  🎯 Train ratio: {train_ratio}")
    console.print()

    if not Path(data_root).exists():
        console.print(
            f"❌ [bold red]Error:[/bold red] Data root directory '{data_root}' does not exist"
        )
        raise typer.Exit(1)

    if not Path(cases_file).exists():
        console.print(
            f"❌ [bold red]Error:[/bold red] Cases file '{cases_file}' does not exist"
        )
        raise typer.Exit(1)

    if not 0 < train_ratio < 1:
        console.print(
            f"❌ [bold red]Error:[/bold red] Train ratio must be between 0 and 1"
        )
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Creating dataset...", total=None)

            total_chunks = create_dataset_streaming(
                data_root=data_root,
                cases_file=cases_file,
                output_dir=output_dir,
                max_cases=max_cases,
                batch_size=batch_size,
                chunk_length=chunk_length,
                train_ratio=train_ratio,
            )

            progress.update(task, completed=True)

        console.print()
        console.print(
            "✅ [bold green]Dataset creation completed successfully![/bold green]"
        )
        console.print(f"📊 Total chunks created: [bold cyan]{total_chunks}[/bold cyan]")
        console.print(f"📁 Output saved to: [bold cyan]{output_dir}[/bold cyan]")

    except KeyboardInterrupt:
        console.print("\n⚠️  [bold yellow]Process interrupted by user[/bold yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(
            f"\n❌ [bold red]Error during dataset creation:[/bold red] {str(e)}"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
