from abc import ABC, abstractmethod

from llm_eval_harness.models import JudgeScore


class BaseJudge(ABC):
    @abstractmethod
    def score(
        self,
        prompt: str,
        output: str,
        expected_output: str | None,
        output_schema: dict,
    ) -> JudgeScore:
        """Score an LLM output and return structured scores."""
