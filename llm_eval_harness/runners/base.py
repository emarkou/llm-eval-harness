from abc import ABC, abstractmethod


class BaseRunner(ABC):
    @abstractmethod
    def run(self, prompt: str) -> str:
        """Send prompt to the target LLM and return the text response."""
