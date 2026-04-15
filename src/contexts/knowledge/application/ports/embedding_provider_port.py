from abc import ABC, abstractmethod


class EmbeddingProviderPort(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError
