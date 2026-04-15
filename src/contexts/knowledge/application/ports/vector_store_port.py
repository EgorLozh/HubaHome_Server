from abc import ABC, abstractmethod


class VectorStorePort(ABC):
    @abstractmethod
    async def upsert(self, collection: str, item_id: str, vector: list[float], payload: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(self, collection: str, vector: list[float], limit: int = 5) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        raise NotImplementedError
