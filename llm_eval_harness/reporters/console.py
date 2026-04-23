from __future__ import annotations

from llm_eval_harness.models import EvalResult

try:
    from rich.console import Console
    from rich.table import Table

    _RICH = True
except ImportError:
    _RICH = False


def _fmt(val: float) -> str:
    return f"{val:.2f}"


def print_results(results: list[EvalResult]) -> None:
    scored = [r for r in results if r.scores is not None]
    failed = [r for r in results if r.error is not None]

    if _RICH:
        _print_rich(results, scored, failed)
    else:
        _print_plain(results, scored, failed)


def _print_rich(
    results: list[EvalResult],
    scored: list[EvalResult],
    failed: list[EvalResult],
) -> None:
    console = Console()

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Correct", justify="center")
    table.add_column("Relevant", justify="center")
    table.add_column("Format", justify="center")
    table.add_column("Reasoning")
    table.add_column("Error", style="red")

    for r in results:
        if r.scores:
            table.add_row(
                r.input_id,
                _fmt(r.scores.correctness),
                _fmt(r.scores.relevance),
                _fmt(r.scores.format_compliance),
                r.scores.reasoning,
                "",
            )
        else:
            table.add_row(r.input_id, "-", "-", "-", "", r.error or "unknown error")

    console.print(table)

    if scored:
        n = len(scored)
        avg_c = sum(r.scores.correctness for r in scored) / n
        avg_r = sum(r.scores.relevance for r in scored) / n
        avg_f = sum(r.scores.format_compliance for r in scored) / n

        summary = Table(title="Summary", show_header=False)
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")
        summary.add_row("Evaluated", str(n))
        summary.add_row("Failed", str(len(failed)))
        summary.add_row("Avg Correctness", _fmt(avg_c))
        summary.add_row("Avg Relevance", _fmt(avg_r))
        summary.add_row("Avg Format Compliance", _fmt(avg_f))
        console.print(summary)


def _print_plain(
    results: list[EvalResult],
    scored: list[EvalResult],
    failed: list[EvalResult],
) -> None:
    header = f"{'ID':<20} {'Correct':>8} {'Relevant':>9} {'Format':>7}  Reasoning"
    print("\n=== Evaluation Results ===")
    print(header)
    print("-" * len(header))

    for r in results:
        if r.scores:
            s = r.scores
            print(
                f"{r.input_id:<20} {_fmt(s.correctness):>8} {_fmt(s.relevance):>9}"
                f" {_fmt(s.format_compliance):>7}  {s.reasoning}"
            )
        else:
            print(f"{r.input_id:<20} {'ERR':>8} {'ERR':>9} {'ERR':>7}  {r.error}")

    if scored:
        n = len(scored)
        avg_c = sum(r.scores.correctness for r in scored) / n
        avg_r = sum(r.scores.relevance for r in scored) / n
        avg_f = sum(r.scores.format_compliance for r in scored) / n

        print("\n=== Summary ===")
        print(f"Evaluated : {n}  |  Failed : {len(failed)}")
        print(
            f"Avg Correctness: {_fmt(avg_c)}  |  "
            f"Avg Relevance: {_fmt(avg_r)}  |  "
            f"Avg Format Compliance: {_fmt(avg_f)}"
        )
