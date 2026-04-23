import openai

from .base import BaseRunner


class OpenAIRunner(BaseRunner):
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self._client = openai.OpenAI()
        self._model = model
        self._max_tokens = max_tokens

    def run(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
