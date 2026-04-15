import httpx

from src.contexts.knowledge.application.ports.vector_store_port import VectorStorePort


class QdrantVectorStoreAdapter(VectorStorePort):
    name = "qdrant"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def upsert(self, collection: str, item_id: str, vector: list[float], payload: dict) -> None:
        return None

    async def search(self, collection: str, vector: list[float], limit: int = 5) -> list[dict]:
        return []

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/collections")
            return response.status_code < 500
        except Exception:
            return False
