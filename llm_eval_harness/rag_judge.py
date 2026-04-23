"""RAG-specific LLM judge.

Scores three dimensions:
  - context_relevance : are the retrieved chunks relevant to the question?
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevance  : does the answer actually address the question?
"""
from __future__ import annotations

import json
import re
from typing import Literal

import anthropic
import openai

from llm_eval_harness.models import DimensionScore, RAGScore

_JUDGE_SYSTEM = """\
You are an expert evaluator of Retrieval-Augmented Generation (RAG) systems. \
Given a question, a list of retrieved context chunks, and the system's answer, \
score the response on three dimensions (0.0–1.0) with brief reasoning for each.

Scoring rubric:
- context_relevance : How well do the retrieved chunks address the question?
    1.0 = all chunks are directly relevant; 0.0 = completely irrelevant.
- faithfulness : Is the answer grounded in the retrieved context, or does it \
    introduce information not present there?
    1.0 = fully supported by context; 0.0 = entirely hallucinated / contradicts context.
- answer_relevance : Does the answer actually respond to the question?
    1.0 = directly and completely answers the question; 0.0 = off-topic or non-answer.

Return ONLY valid JSON (no markdown fences) matching this exact structure:
{
  "context_relevance": {"score": <float>, "reasoning": "<1-2 sentences>"},
  "faithfulness":      {"score": <float>, "reasoning": "<1-2 sentences>"},
  "answer_relevance":  {"score": <float>, "reasoning": "<1-2 sentences>"}
}"""


def _build_user_message(
    question: str,
    retrieved_texts: list[str],
    answer: str,
) -> str:
    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{text}" for i, text in enumerate(retrieved_texts)
    )
    return "\n\n".join([
        f"## Question\n{question}",
        f"## Retrieved context chunks\n{context_block}",
        f"## System answer\n{answer}",
    ])


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


class RAGJudge:
    """Judges a RAG response using a configurable LLM as evaluator."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"],
        model: str,
        max_tokens: int = 768,
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
        question: str,
        retrieved_texts: list[str],
        answer: str,
    ) -> RAGScore:
        user_msg = _build_user_message(question, retrieved_texts, answer)

        if self._provider == "anthropic":
            raw = self._call_anthropic(user_msg)
        else:
            raw = self._call_openai(user_msg)

        data = json.loads(_strip_fences(raw))
        return RAGScore(
            context_relevance=DimensionScore(**data["context_relevance"]),
            faithfulness=DimensionScore(**data["faithfulness"]),
            answer_relevance=DimensionScore(**data["answer_relevance"]),
        )

    def _call_anthropic(self, user_msg: str) -> str:
        message = self._anthropic.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text

    def _call_openai(self, user_msg: str) -> str:
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
