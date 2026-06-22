"""
NUR V3 Runtime — CLI

Usage:
  python -m nur.v3.cli "your question here"
  python -m nur.v3.cli  # interactive REPL

Flags:
  --lang fr|en|ar   Force output language (default: auto-detect)
  --verbose         Show all intermediate steps
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from nur.v3.pipeline import run_pipeline, _detect_language

app = typer.Typer(help="NUR V3 — Islamic RAG CLI")
console = Console()


def _print_result(result, verbose: bool = False):
    """Pretty-print a PipelineResult."""
    # Confidence panel
    conf_color = {"STRONG": "green", "WEAK": "yellow", "EMPTY": "red"}[result.phase_a_status]
    console.print(Panel.fit(
        f"Phase A (Quran): [{conf_color}]{result.phase_a_status}[/{conf_color}] "
        f"(confidence={result.phase_a_confidence:.2f})\n"
        f"Phase B (Hadith): [{conf_color}]{result.phase_b_status}[/{conf_color}] "
        f"(confidence={result.phase_b_confidence:.2f})",
        title="Confidence",
        border_style="blue",
    ))

    # Answer
    answer = result.response.get("answer", "(no answer)")
    console.print(Panel(answer, title="Answer", border_style="cyan"))

    # Citations
    citations = result.response.get("citations", [])
    if citations:
        table = Table(title="Citations", show_lines=True)
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Arabic", style="yellow")
        table.add_column("URL", style="blue")
        for cit in citations:
            ar = cit.get("arabic", "")[:60] + "..." if len(cit.get("arabic", "")) > 60 else cit.get("arabic", "")
            table.add_row(
                cit.get("label", cit.get("source_id", "?")),
                cit.get("type", "?"),
                ar,
                cit.get("url", ""),
            )
        console.print(table)

    # Ikhtilaf
    ikh = result.response.get("ikhtilaf", {})
    if ikh.get("detected"):
        console.print(Panel(
            f"⚠️ Ikhtilaf detected:\n{ikh.get('summary', '')}",
            title="Ikhtilaf (Disagreement)",
            border_style="yellow",
        ))

    # Disclaimer
    disc = result.response.get("disclaimer")
    if disc:
        console.print(Panel(disc, title="Disclaimer", border_style="dim"))

    # Verification
    if result.verification_errors:
        err_table = Table(title="Verification issues", show_lines=True)
        err_table.add_column("Issue", style="red")
        err_table.add_column("Detail", style="yellow")
        for e in result.verification_errors:
            err_table.add_row(e.get("issue", "?"), str({k: v for k, v in e.items() if k != "issue"}))
        console.print(err_table)
    else:
        console.print("[green]✅ All verification checks passed[/green]")

    if verbose:
        console.print(f"\n[dim]Total time: {result.total_seconds:.1f}s[/dim]")
        for step, sec in result.step_timings.items():
            console.print(f"[dim]  {step}: {sec:.1f}s[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(None, help="Question to ask NUR"),
    lang: str = typer.Option(None, "--lang", "-l", help="Force output language (fr/en/ar)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show intermediate steps"),
):
    """Ask NUR a single question."""
    if not question:
        question = Prompt.ask("[bold cyan]Ask NUR[/bold cyan]")

    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("NUR_GROQ_API_KEY"):
        console.print("[red]ERROR: GROQ_API_KEY environment variable required[/red]")
        console.print("Get a free key at https://console.groq.com")
        raise typer.Exit(1)

    forced_lang = lang
    if forced_lang:
        detected_lang = forced_lang
    else:
        detected_lang = _detect_language(question)

    with console.status("[bold cyan]Running NUR V3 pipeline...[/bold cyan]"):
        result = run_pipeline(question)

    _print_result(result, verbose=verbose)


@app.command()
def repl(
    lang: str = typer.Option(None, "--lang", "-l"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Interactive REPL mode."""
    console.print(Panel.fit(
        "[bold cyan]NUR V3 — Islamic RAG[/bold cyan]\n"
        "Type 'exit' or 'quit' to leave.",
        border_style="cyan",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Question[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        if question.strip().lower() in ("exit", "quit", "q"):
            break
        if not question.strip():
            continue

        with console.status("[bold cyan]Running pipeline...[/bold cyan]"):
            result = run_pipeline(question)

        _print_result(result, verbose=verbose)


if __name__ == "__main__":
    app()
