from abc import ABC, abstractmethod


class LLMProviderPort(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        raise NotImplementedError
