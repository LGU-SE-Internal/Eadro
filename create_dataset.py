#!/usr/bin/env python3
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich import print as rprint
from src.eadro.data_adapter import create_dataset
from src.eadro.progress_manager import ProgressManager

app = typer.Typer(
    help="Create and manage datasets with incremental processing for EADRO"
)
console = Console()


@app.command()
def main(
    data_root: str = typer.Argument(
        "/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/",
        help="Root directory containing data packs",
    ),
    cases_file: str = typer.Argument(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet",
        help="Path to cases parquet file",
    ),
    output_dir: str = typer.Argument(
        "dataset_output", help="Output directory for processed dataset"
    ),
    max_cases: Optional[int] = typer.Option(
        None, "--max-cases", help="Maximum number of cases to process"
    ),
    chunk_length: int = typer.Option(10, "--chunk-length", help="Length of each chunk"),
    train_ratio: float = typer.Option(
        0.7, "--train-ratio", help="Ratio of training data", min=0.0, max=1.0
    ),
    n_workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of worker processes (default: CPU count)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    if not Path(data_root).exists():
        rprint(f"[red]Error: Data root directory '{data_root}' does not exist[/red]")
        raise typer.Exit(1)

    if not Path(cases_file).exists():
        rprint(f"[red]Error: Cases file '{cases_file}' does not exist[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Dataset Creation Configuration[/bold blue]")
    console.print(f"  [cyan]Data root:[/cyan] {data_root}")
    console.print(f"  [cyan]Cases file:[/cyan] {cases_file}")
    console.print(f"  [cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"  [cyan]Max cases:[/cyan] {max_cases or 'All'}")
    console.print(f"  [cyan]Chunk length:[/cyan] {chunk_length}")
    console.print(f"  [cyan]Train ratio:[/cyan] {train_ratio}")
    console.print(f"  [cyan]Workers:[/cyan] {n_workers or 'Auto'}")
    console.print()

    if not typer.confirm("Continue with dataset creation?"):
        rprint("[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(0)

    try:
        progress_manager = ProgressManager(console)
        console.print(
            "\n[bold blue]Starting dataset creation with Docker-style progress display...[/bold blue]"
        )
        progress_manager.start()

        try:
            total_chunks = create_dataset(
                data_root=data_root,
                cases_file=cases_file,
                output_dir=output_dir,
                max_cases=max_cases,
                chunk_length=chunk_length,
                train_ratio=train_ratio,
                n_workers=n_workers,
                progress_manager=progress_manager,
            )

            progress_manager.wait_for_completion(timeout=10.0)

        finally:
            progress_manager.stop()

        summary = progress_manager.get_summary()
        console.print(
            "\n[bold green]✓ Dataset creation completed successfully![/bold green]"
        )
        console.print(f"[green]Total chunks created: {total_chunks}[/green]")
        console.print(f"[green]Dataset saved to: {output_dir}[/green]")
        console.print("\n[bold blue]Task Summary:[/bold blue]")
        console.print(f"  [green]✅ Completed: {summary['completed']}[/green]")
        console.print(f"  [red]❌ Failed: {summary['failed']}[/red]")
        console.print(f"  [yellow]⏳ Pending: {summary['pending']}[/yellow]")
        console.print(f"  [blue]🔄 Running: {summary['running']}[/blue]")

    except KeyboardInterrupt:
        rprint("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print("\n[bold red]✗ Error during dataset creation:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
