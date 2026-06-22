"""
cli.py

This file is the user-facing terminal interface for NUR. It wraps the NURPipeline
in a beautiful, interactive Rich + Typer CLI that displays answers with:
  - Answer warnings (CRITICAL/BIG/SMALL) shown FIRST, prominently
  - The synthesis (main answer) in a clear panel
  - Each cited source as a sub-panel with Arabic text (large, RTL), translation,
    grade explanation (for hadiths), and a clickable URL
  - Sub-questions at the bottom for transparency
  - Timing information

WHY THIS EXISTS:
  The pipeline returns a PipelineResult dataclass with all intermediate data.
  But that dataclass is for machines — the CLI is for humans. It translates the
  structured output into a beautiful, readable, theologically-safe display that
  respects every pillar:
    - Pillar 4 (Absolute Reliability): warnings shown first, before the answer
    - Pillar 7 (Structured Citation): every source has a clickable URL
    - Pillar 10 (Arabic Source of Truth): Arabic text always visible, large, RTL

TWO MODES:
  1. Single query:  python -m nur.cli "What does the Quran say about charity?"
  2. Interactive:   python -m nur.cli
     Then type questions at the NUR> prompt. Type 'exit' or 'quit' to leave.

FLAGS:
  --lang fr          Force French synthesis (default: English, auto-detects)
  --force-reasoning  Use the Llama 3.3 70B model instead of Scout (for extreme
                     Ikhtilaf cases). Consumes 12K TPM instead of 30K TPM.
  --no-arabic        Hide Arabic text (default: show). For terminals without
                     Arabic font support.
  --verbose          Show sub-questions and retrieval details (default: hide
                     in interactive mode, show in single-query mode).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .config import settings
from .pipeline import NURPipeline, PipelineResult

# ============================================================
# Console setup — RTL support for Arabic, wide output
# ============================================================

# We create one Console instance and reuse it. `rtl=False` on the console
# itself (LTR for English UI), but we render Arabic text blocks with
# `justify="right"` and a larger style so they stand out.
console = Console(width=100)

# ============================================================
# Display functions — each renders one part of the result
# ============================================================


def display_answer_warning(result: PipelineResult) -> None:
    """Display the answer-level warning in a prominent colored panel.

    Per Pillar 4 (Absolute Reliability), the warning MUST appear BEFORE the
    answer so the user sees it before reading any content. The panel color
    reflects severity:
      - CRITICAL (Mawdu'): red background
      - BIG (only Da'if): red border
      - SMALL (Da'if alongside Sahih/Hasan): yellow border
      - NONE: no panel shown
    """
    if not result.answer_warning:
        return

    # Determine severity by checking the warning text prefix
    if result.answer_warning.startswith("🚫"):
        # CRITICAL — fabricated hadith
        console.print()
        console.print(Panel(
            Text(result.answer_warning, style="bold white on red"),
            border_style="red",
            title="[bold red]CRITICAL WARNING",
            title_align="left",
        ))
    elif "IMPORTANT" in result.answer_warning:
        # BIG — no authentic sources
        console.print()
        console.print(Panel(
            Text(result.answer_warning, style="bold yellow"),
            border_style="red",
            title="[bold red]⚠️  WARNING",
            title_align="left",
        ))
    else:
        # SMALL — additional context note
        console.print()
        console.print(Panel(
            Text(result.answer_warning, style="yellow"),
            border_style="yellow",
            title="[yellow]ℹ️  Note",
            title_align="left",
        ))


def display_synthesis(result: PipelineResult) -> None:
    """Display the Reporter's synthesis (the main answer) in a panel.

    The synthesis is the final assembled answer that cites source IDs like
    [S1], [S3]. We render it as rich text so the citations stand out.
    """
    if not result.report:
        return

    console.print()
    console.print(Panel(
        result.report.synthesis,
        title="[bold cyan]📜 Answer",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    ))


def display_conflict_detection(result: PipelineResult) -> None:
    """Display the conflict detection (Ikhtilaf awareness) if non-trivial.

    Per Pillar 9, when scholars disagree, NUR presents all views neutrally.
    The conflict_detection field tells the user whether disagreement exists.
    We only show this panel if a conflict was detected (not the default
    "No conflict detected" message).
    """
    if not result.report:
        return

    conflict_text = result.report.conflict_detection
    # Don't show the panel if there's no conflict — keep the output clean
    if "no conflict" in conflict_text.lower() or "consistent" in conflict_text.lower():
        return

    console.print()
    console.print(Panel(
        conflict_text,
        title="[bold magenta]⚖️  Scholarly Disagreement (Ikhtilaf)",
        title_align="left",
        border_style="magenta",
        padding=(1, 2),
    ))


def display_direct_reports(
    result: PipelineResult,
    show_arabic: bool = True,
) -> None:
    """Display each cited source as a sub-panel with Arabic + translation + URL.

    Per Pillar 10, Arabic text is ALWAYS visible (unless --no-arabic is passed).
    Per Pillar 7, every source has a clickable URL.

    Args:
        result: The PipelineResult from the pipeline.
        show_arabic: If True, render Arabic text in a large RTL style. If False,
                     skip Arabic (for terminals without Arabic font support).
    """
    if not result.report or not result.report.direct_reports:
        return

    # Build a lookup map: source_id (S1, S2, ...) → SourceRef
    source_map = {f"S{i+1}": ref for i, ref in enumerate(result.top_chunks)}

    console.print()
    console.print(Rule("[bold blue]📚 Sources", style="blue"))

    for report in result.report.direct_reports:
        ref = source_map.get(report.source_id)
        if not ref:
            continue

        # Build the panel content line by line
        lines: list[str] = []

        # Source ID + label + URL (clickable in modern terminals)
        url_text = ref.url if ref.url else "(no URL)"
        lines.append(f"[bold]{report.source_id}[/bold] — {ref.display_label}")
        lines.append(f"[link={url_text}]{url_text}[/link]" if ref.url else url_text)

        # Grade (for hadiths only — Quran and Tafsir don't have grades)
        if ref.kind == "hadith" and ref.grade:
            grade_display = ref.grade
            lines.append(f"[dim]Grade:[/dim] {grade_display}")

        # Arabic text — ALWAYS shown per Pillar 10 (unless --no-arabic)
        if show_arabic and report.arabic_text:
            lines.append("")
            lines.append("[bold]العربية:[/bold]")
            lines.append(report.arabic_text)

        # English/translation text
        if report.report:
            lines.append("")
            lines.append(f"[dim]Report:[/dim] {report.report}")

        # Grade explanation (education layer from DEC-027)
        if report.source_id in result.grade_explanations:
            explanation = result.grade_explanations[report.source_id]
            lines.append("")
            lines.append(f"[dim]📚 {explanation}[/dim]")

        panel_content = "\n".join(lines)

        # Choose border color based on source type
        border_color = {
            "quran": "green",
            "hadith": "blue",
            "tafsir_ar": "magenta",
            "tafsir_en": "magenta",
            "scholar": "yellow",
        }.get(ref.kind, "white")

        console.print()
        console.print(Panel(
            panel_content,
            border_style=border_color,
            padding=(1, 2),
            expand=True,
        ))


def display_sub_questions(result: PipelineResult, verbose: bool) -> None:
    """Display the Architect's sub-questions (transparency layer).

    Only shown in verbose mode (default for single-query, off for interactive).
    This lets the user see how their question was decomposed.
    """
    if not verbose or not result.sub_questions:
        return

    console.print()
    console.print(Rule("[dim]🔍 Query Decomposition (Architect)", style="dim"))

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Num", style="dim", width=4)
    table.add_column("Sub-question", style="dim")

    for i, sq in enumerate(result.sub_questions, 1):
        table.add_row(f"{i}.", sq)

    console.print(table)


def display_retrieval_summary(result: PipelineResult, verbose: bool) -> None:
    """Display a brief summary of what was retrieved.

    Only shown in verbose mode. Shows the top 5 chunks by RRF score with
    their source and ranks, so the user can see what the retriever found.
    """
    if not verbose or not result.retrieved_chunks:
        return

    console.print()
    console.print(Rule("[dim]📊 Retrieval Summary (top 5)", style="dim"))

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Chunk ID", style="dim")
    table.add_column("Source", style="dim", width=10)
    table.add_column("RRF Score", style="dim", width=12)

    for i, chunk in enumerate(result.retrieved_chunks[:5], 1):
        table.add_row(
            str(i),
            chunk["id"],
            chunk["source"],
            f"{chunk['rrf_score']:.6f}",
        )

    console.print(table)


def display_error(result: PipelineResult) -> None:
    """Display an error panel if the pipeline failed."""
    if not result.error:
        return

    console.print()
    console.print(Panel(
        Text(result.error, style="bold white on red"),
        title="[bold red]❌ Error",
        title_align="left",
        border_style="red",
    ))


def display_result(
    result: PipelineResult,
    show_arabic: bool = True,
    verbose: bool = False,
) -> None:
    """Display the complete pipeline result in the correct order.

    The order is deliberate and pillar-compliant:
      1. Answer warning (Pillar 4 — must be seen first)
      2. Conflict detection (Pillar 9 — Ikhtilaf awareness)
      3. Synthesis (the main answer)
      4. Direct reports with Arabic + URLs + grades (Pillars 7, 10, 3)
      5. Sub-questions (transparency, verbose only)
      6. Retrieval summary (transparency, verbose only)
      7. Error (if any)
    """
    # Header
    console.print()
    console.print(Rule(f"[bold green]نور (NUR) — Light", style="green"))

    # 1. Answer warning (if any)
    display_answer_warning(result)

    # 2. Conflict detection (if non-trivial)
    display_conflict_detection(result)

    # 3. Synthesis (main answer)
    display_synthesis(result)

    # 4. Direct reports
    display_direct_reports(result, show_arabic=show_arabic)

    # 5. Sub-questions (verbose only)
    display_sub_questions(result, verbose=verbose)

    # 6. Retrieval summary (verbose only)
    display_retrieval_summary(result, verbose=verbose)

    # 7. Error (if any)
    display_error(result)

    console.print()


# ============================================================
# Pipeline singleton — lazy-loaded on first use
# ============================================================

_pipeline: NURPipeline | None = None


def get_pipeline() -> NURPipeline:
    """Get or create the singleton NURPipeline instance.

    The pipeline is expensive to construct (loads BGE-M3 on first query).
    We cache it so subsequent queries in interactive mode are fast.
    """
    global _pipeline
    if _pipeline is None:
        console.print("[dim]Initializing NUR pipeline...[/dim]")
        start = time.time()
        _pipeline = NURPipeline()
        elapsed = time.time() - start
        console.print(f"[dim]✅ Pipeline ready in {elapsed:.1f}s (BGE-M3 on {_pipeline._device})[/dim]")
        console.print()
    return _pipeline


# ============================================================
# Typer app — the CLI entry point
# ============================================================

app = typer.Typer(
    name="nur",
    help="NUR (نور) — Islamic RAG chatbot. Ask any question about Islam.",
    no_args_is_help=False,
    add_completion=False,
)


@app.command()
def main(
    query: str = typer.Argument(
        None,
        help="Your question about Islam. If omitted, enters interactive mode.",
    ),
    lang: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="Synthesis language: 'en' (default) or 'fr'. Arabic is always shown.",
    ),
    force_reasoning: bool = typer.Option(
        False,
        "--force-reasoning",
        "-r",
        help="Use Llama 3.3 70B instead of Scout (for extreme Ikhtilaf).",
    ),
    no_arabic: bool = typer.Option(
        False,
        "--no-arabic",
        help="Hide Arabic text (for terminals without Arabic font support).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show sub-questions and retrieval details.",
    ),
) -> None:
    """Ask NUR a question about Islam, or enter interactive mode.

    Examples:
      python -m nur.cli "What does the Quran say about charity?"
      python -m nur.cli --lang fr "Quel est le statut de l'usure?"
      python -m nur.cli  # interactive mode
    """
    # Pre-flight: check API key
    if not settings.groq_api_key:
        console.print("[bold red]❌ GROQ_API_KEY is not set. Add it to .env and retry.[/bold red]")
        raise typer.Exit(1)

    # If a query is provided, answer it and exit (single-query mode)
    if query:
        _run_single_query(
            query=query,
            force_reasoning=force_reasoning,
            show_arabic=not no_arabic,
            verbose=verbose,
        )
        return

    # No query → interactive REPL mode
    _run_interactive(
        force_reasoning=force_reasoning,
        show_arabic=not no_arabic,
        verbose=verbose,
    )


def _run_single_query(
    query: str,
    force_reasoning: bool,
    show_arabic: bool,
    verbose: bool,
) -> None:
    """Run a single query and display the result, then exit."""
    pipeline = get_pipeline()

    console.print(f"[dim]Question: {query}[/dim]")
    start = time.time()
    result = pipeline.query(query, force_reasoning=force_reasoning)
    elapsed = time.time() - start

    display_result(result, show_arabic=show_arabic, verbose=verbose)

    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")


def _run_interactive(
    force_reasoning: bool,
    show_arabic: bool,
    verbose: bool,
) -> None:
    """Run the interactive REPL: prompt for questions, answer, repeat.

    Type 'exit', 'quit', or Ctrl+C to leave.
    """
    console.print()
    console.print(Panel(
        "[bold green]NUR (نور)[/bold green] — Islamic RAG Chatbot\n"
        "[dim]Ask any question about Islam. Type 'exit' to quit.[/dim]",
        border_style="green",
        padding=(1, 2),
    ))

    # Initialize the pipeline once (loads BGE-M3 on first query)
    pipeline = get_pipeline()

    while True:
        try:
            # Prompt for the next question
            console.print()
            question = Prompt.ask("[bold green]NUR>[/bold green]")

            # Handle exit commands
            if question.strip().lower() in ("exit", "quit", ":q"):
                console.print("[dim]Peace be upon you. Goodbye.[/dim]")
                break

            # Skip empty input
            if not question.strip():
                continue

            # Run the pipeline
            start = time.time()
            result = pipeline.query(question, force_reasoning=force_reasoning)
            elapsed = time.time() - start

            # Display the result
            display_result(result, show_arabic=show_arabic, verbose=verbose)

            console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")

        except KeyboardInterrupt:
            console.print()
            console.print("[dim]Peace be upon you. Goodbye.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {type(e).__name__}: {e}[/bold red]")
            console.print("[dim]You can ask another question or type 'exit'.[/dim]")


# ============================================================
# Entry point — allows `python src/nur/cli.py` AND `python -m nur.cli`
# ============================================================

if __name__ == "__main__":
    app()
