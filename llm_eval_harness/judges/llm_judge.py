from __future__ import annotations

import json
import re
from typing import Literal

import anthropic
import openai

from llm_eval_harness.models import JudgeScore

from .base import BaseJudge

_JUDGE_SYSTEM = """\
You are an impartial evaluator of LLM outputs. Given the original prompt, the \
model's response, an optional expected answer, and an output schema, score the \
response on three dimensions (0.0–1.0 floats) and provide brief reasoning.

Scoring rubric:
- correctness: factual accuracy compared to the expected answer (or general knowledge if none provided)
- relevance: how well the response addresses what was asked
- format_compliance: how well the response matches the specified output schema/format

Return ONLY valid JSON matching this structure (no markdown fences):
{
  "correctness": <float>,
  "relevance": <float>,
  "format_compliance": <float>,
  "reasoning": "<one or two sentences>"
}"""


def _build_user_message(
    prompt: str,
    output: str,
    expected_output: str | None,
    output_schema: dict,
) -> str:
    parts = [
        f"## Original prompt\n{prompt}",
        f"## Model output\n{output}",
        f"## Expected output schema\n{json.dumps(output_schema, indent=2)}",
    ]
    if expected_output:
        parts.insert(2, f"## Expected answer\n{expected_output}")
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes add despite instructions."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


class LLMJudge(BaseJudge):
    def __init__(
        self,
        provider: Literal["openai", "anthropic"],
        model: str,
        max_tokens: int = 512,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

        if provider == "anthropic":
            self._anthropic = anthropic.Anthropic()
        else:
            self._openai = openai.OpenAI()

    def score(
        self,
        prompt: str,
        output: str,
        expected_output: str | None,
        output_schema: dict,
    ) -> JudgeScore:
        user_msg = _build_user_message(prompt, output, expected_output, output_schema)

        if self._provider == "anthropic":
            raw = self._score_anthropic(user_msg)
        else:
            raw = self._score_openai(user_msg)

        data = json.loads(_strip_fences(raw))
        return JudgeScore(**data)

    def _score_anthropic(self, user_msg: str) -> str:
        message = self._anthropic.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text

    def _score_openai(self, user_msg: str) -> str:
        response = self._openai.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content or "{}"
