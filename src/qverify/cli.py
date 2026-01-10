"""
Command-line interface for QVERIFY.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(verbose: bool, debug: bool) -> None:
    """QVERIFY: LLM-Assisted Quantum Program Verification."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


@main.command()
@click.argument('source', type=click.Path(exists=True))
@click.option('--llm', '-l', default='claude-3.5-sonnet', help='LLM model to use')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--language', type=click.Choice(['silq', 'openqasm']), default='silq')
def verify(source: str, llm: str, output: Optional[str], language: str) -> None:
    """Verify a quantum program."""
    from qverify import QVerify
    
    console.print(f"[bold blue]QVERIFY[/bold blue] - Verifying {source}")
    console.print(f"Using LLM: {llm}")
    
    try:
        qv = QVerify(llm=llm)
        result = qv.verify_file(Path(source))
        
        if result.is_valid():
            console.print("[bold green]✓ Verification PASSED[/bold green]")
        elif result.is_invalid():
            console.print("[bold red]✗ Verification FAILED[/bold red]")
            if result.counterexample:
                console.print(f"Counterexample: {result.counterexample}")
        else:
            console.print("[bold yellow]? Verification UNKNOWN[/bold yellow]")
        
        console.print(f"Time: {result.time_seconds:.2f}s")
        
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "status": result.status.name,
                    "message": result.message,
                    "time": result.time_seconds,
                }, f, indent=2)
            console.print(f"Results saved to {output}")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument('source', type=click.Path(exists=True))
@click.option('--llm', '-l', default='claude-3.5-sonnet', help='LLM model to use')
@click.option('--output', '-o', type=click.Path(), help='Output file for specification')
def synthesize(source: str, llm: str, output: Optional[str]) -> None:
    """Synthesize specification for a quantum program."""
    from qverify import QVerify, QuantumProgram
    
    console.print(f"[bold blue]QVERIFY[/bold blue] - Synthesizing specification for {source}")
    
    try:
        qv = QVerify(llm=llm)
        program = QuantumProgram.from_file(Path(source))
        result = qv.synthesize_specification(program)
        
        if result.is_success():
            console.print("[bold green]✓ Synthesis SUCCEEDED[/bold green]")
            console.print(f"\n{result.specification}")
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result.specification.to_dict(), f, indent=2)
                console.print(f"\nSpecification saved to {output}")
        else:
            console.print(f"[bold red]✗ Synthesis FAILED[/bold red]: {result.message}")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.option('--tier', '-t', type=click.Choice(['T1', 'T2', 'T3', 'T4', 'T5', 'all']), default='T2')
@click.option('--llm', '-l', default='claude-3.5-sonnet', help='LLM model to evaluate')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--timeout', default=60.0, help='Timeout per program')
def benchmark(tier: str, llm: str, output: Optional[str], timeout: float) -> None:
    """Run QVerifyBench evaluation."""
    from qverify.benchmark import QVerifyBench
    
    console.print(f"[bold blue]QVERIFY Benchmark[/bold blue]")
    console.print(f"Tier: {tier}, LLM: {llm}")
    
    bench = QVerifyBench(tier=None if tier == 'all' else tier)
    console.print(f"Programs: {len(bench)}")
    
    with Progress() as progress:
        task = progress.add_task("Evaluating...", total=len(bench))
        results = bench.evaluate(llm=llm, timeout=timeout)
        progress.update(task, completed=len(bench))
    
    # Display results table
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Synthesis Rate", f"{results.synthesis_rate:.1%}")
    table.add_row("Verification Rate", f"{results.verification_rate:.1%}")
    table.add_row("Accuracy", f"{results.accuracy:.1%}")
    table.add_row("Avg Time", f"{results.avg_time:.2f}s")
    table.add_row("Total Time", f"{results.total_time_seconds:.2f}s")
    
    console.print(table)
    
    if output:
        bench.save_results(results, Path(output))
        console.print(f"Results saved to {output}")


@main.command()
def info() -> None:
    """Display QVERIFY information."""
    from qverify import __version__
    
    console.print(f"[bold blue]QVERIFY[/bold blue] v{__version__}")
    console.print("LLM-Assisted Formal Verification of Quantum Programs")
    console.print("\nComponents:")
    console.print("  • QuantumSpecSynth - Specification synthesis")
    console.print("  • NeuralSilVer - SMT-based verification")
    console.print("  • QVerifyBench - Benchmark framework")
    console.print("\nFor more info: https://github.com/hmshujaatzaheer/qverify-framework")


if __name__ == "__main__":
    main()
