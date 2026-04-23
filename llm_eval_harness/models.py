from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalInput(BaseModel):
    id: str
    variables: dict[str, str]
    expected_output: str | None = None


class EvalCase(BaseModel):
    id: str
    question: str
    expected: str | None = None


class EvalSuite(BaseModel):
    suite_name: str
    description: str = ""
    prompt_template: str
    output_schema: dict[str, Any]
    cases: list[EvalCase]

    def to_eval_inputs(self) -> list[EvalInput]:
        return [
            EvalInput(
                id=case.id,
                variables={"question": case.question},
                expected_output=case.expected,
            )
            for case in self.cases
        ]


class JudgeScore(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    format_compliance: float = Field(ge=0.0, le=1.0)
    reasoning: str


class EvalResult(BaseModel):
    input_id: str
    prompt: str
    raw_output: str
    scores: JudgeScore | None = None
    error: str | None = None


class EvalConfig(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str
    judge_provider: Literal["openai", "anthropic"]
    judge_model: str
    max_tokens: int = 1024
