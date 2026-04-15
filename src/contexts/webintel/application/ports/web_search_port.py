from abc import ABC, abstractmethod


class WebSearchPort(ABC):
    @abstractmethod
    async def search(self, query: str) -> list[dict]:
        raise NotImplementedError
