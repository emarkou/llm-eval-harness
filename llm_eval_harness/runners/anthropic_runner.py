import anthropic

from .base import BaseRunner


class AnthropicRunner(BaseRunner):
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def run(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
