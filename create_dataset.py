#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from src.eadro.data_adapter import create_dataset_streaming

app = typer.Typer(help="Create dataset with parallel processing for EADRO")
console = Console()


@app.command()
def main(
    data_root: str = typer.Argument("/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/", help="Root directory containing data packs"),
    cases_file: str = typer.Argument("/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet", help="Path to cases parquet file"),
    output_dir: str = typer.Argument("dataset_output", help="Output directory for processed dataset"),
    max_cases: Optional[int] = typer.Option(None, "--max-cases", help="Maximum number of cases to process"),
    chunk_length: int = typer.Option(10, "--chunk-length", help="Length of each chunk"),
    train_ratio: float = typer.Option(0.7, "--train-ratio", help="Ratio of training data", min=0.0, max=1.0),
    n_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of worker processes (default: CPU count)"),
    no_parallel: bool = typer.Option(False, "--no-parallel", help="Disable parallel processing"),
    batch_size: int = typer.Option(2, "--batch-size", help="Batch size for sequential processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    if not Path(data_root).exists():
        rprint(f"[red]Error: Data root directory '{data_root}' does not exist[/red]")
        raise typer.Exit(1)
    
    if not Path(cases_file).exists():
        rprint(f"[red]Error: Cases file '{cases_file}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Display configuration
    console.print("\n[bold blue]Dataset Creation Configuration[/bold blue]")
    console.print(f"  [cyan]Data root:[/cyan] {data_root}")
    console.print(f"  [cyan]Cases file:[/cyan] {cases_file}")
    console.print(f"  [cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"  [cyan]Max cases:[/cyan] {max_cases or 'All'}")
    console.print(f"  [cyan]Chunk length:[/cyan] {chunk_length}")
    console.print(f"  [cyan]Train ratio:[/cyan] {train_ratio}")
    console.print(f"  [cyan]Workers:[/cyan] {n_workers or 'Auto'}")
    console.print(f"  [cyan]Parallel processing:[/cyan] {'Disabled' if no_parallel else 'Enabled'}")
    console.print(f"  [cyan]Batch size:[/cyan] {batch_size}")
    console.print()
    
    if not typer.confirm("Continue with dataset creation?"):
        rprint("[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(0)
    
    try:
        with console.status("[bold green]Creating dataset...") as status:
            total_chunks = create_dataset_streaming(
                data_root=data_root,
                cases_file=cases_file,
                output_dir=output_dir,
                max_cases=max_cases,
                batch_size=batch_size,
                chunk_length=chunk_length,
                train_ratio=train_ratio,
                n_workers=n_workers,
                use_parallel=not no_parallel,
            )
        
        console.print(f"\n[bold green]✓ Dataset creation completed successfully![/bold green]")
        console.print(f"[green]Total chunks created: {total_chunks}[/green]")
        console.print(f"[green]Dataset saved to: {output_dir}[/green]")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during dataset creation:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)



if __name__ == "__main__":
    app()
