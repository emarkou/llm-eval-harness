from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from llm_eval_harness.judges.llm_judge import LLMJudge
from llm_eval_harness.models import EvalConfig, EvalInput, EvalResult, EvalSuite
from llm_eval_harness.reporters.console import print_results
from llm_eval_harness.runners import AnthropicRunner, OpenAIRunner
from llm_eval_harness.runners.base import BaseRunner

app = typer.Typer(help="llm-eval-harness: evaluate LLM application outputs")


@app.callback()
def _main() -> None:
    pass


def _make_runner(config: EvalConfig) -> BaseRunner:
    if config.provider == "anthropic":
        return AnthropicRunner(model=config.model, max_tokens=config.max_tokens)
    return OpenAIRunner(model=config.model, max_tokens=config.max_tokens)


@app.command()
def run(
    inputs: Annotated[
        Path,
        typer.Option("--inputs", "-i", help="Path to JSON file: EvalSuite object or list of EvalInput"),
    ],
    template: Annotated[
        Path | None,
        typer.Option("--template", "-t", help="Prompt template file (overrides suite's prompt_template)"),
    ] = None,
    schema: Annotated[
        Path | None,
        typer.Option("--schema", "-s", help="JSON Schema file (overrides suite's output_schema)"),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Target LLM provider: anthropic or openai"),
    ] = "anthropic",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Target LLM model name"),
    ] = "claude-haiku-4-5-20251001",
    judge_provider: Annotated[
        str,
        typer.Option("--judge-provider", help="Judge LLM provider: anthropic or openai"),
    ] = "anthropic",
    judge_model: Annotated[
        str,
        typer.Option("--judge-model", help="Judge LLM model name"),
    ] = "claude-haiku-4-5-20251001",
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max tokens for target LLM responses"),
    ] = 1024,
) -> None:
    """Run an evaluation suite and print a scored report to stdout."""
    config = EvalConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        judge_provider=judge_provider,  # type: ignore[arg-type]
        judge_model=judge_model,
        max_tokens=max_tokens,
    )

    raw = json.loads(inputs.read_text())

    if isinstance(raw, dict) and "cases" in raw:
        suite = EvalSuite(**raw)
        prompt_template = template.read_text() if template else suite.prompt_template
        output_schema: dict = json.loads(schema.read_text()) if schema else suite.output_schema
        eval_inputs = suite.to_eval_inputs()
        typer.echo(f"Suite: {suite.suite_name} — {suite.description}", err=True)
    else:
        if template is None or schema is None:
            typer.echo("ERROR: --template and --schema are required when inputs is not an EvalSuite.", err=True)
            raise typer.Exit(code=1)
        prompt_template = template.read_text()
        output_schema = json.loads(schema.read_text())
        eval_inputs = [EvalInput(**item) for item in raw]

    runner = _make_runner(config)
    judge = LLMJudge(
        provider=config.judge_provider,
        model=config.judge_model,
        max_tokens=512,
    )

    results: list[EvalResult] = []
    total = len(eval_inputs)

    for idx, eval_input in enumerate(eval_inputs, start=1):
        typer.echo(f"[{idx}/{total}] Running input '{eval_input.id}'...", err=True)
        try:
            rendered_prompt = prompt_template.format(**eval_input.variables)
            raw_output = runner.run(rendered_prompt)
            scores = judge.score(
                prompt=rendered_prompt,
                output=raw_output,
                expected_output=eval_input.expected_output,
                output_schema=output_schema,
            )
            results.append(
                EvalResult(
                    input_id=eval_input.id,
                    prompt=rendered_prompt,
                    raw_output=raw_output,
                    scores=scores,
                )
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  ERROR: {exc}", err=True)
            results.append(
                EvalResult(
                    input_id=eval_input.id,
                    prompt=prompt_template,
                    raw_output="",
                    error=str(exc),
                )
            )

    print_results(results)
