import httpx

from src.contexts.knowledge.application.ports.vector_store_port import VectorStorePort


class QdrantVectorStoreAdapter(VectorStorePort):
    name = "qdrant"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._known_collections: set[str] = set()

    async def upsert(self, collection: str, item_id: str, vector: list[float], payload: dict) -> None:
        if not vector:
            return None

        async with httpx.AsyncClient(timeout=5.0) as client:
            await self._ensure_collection(client=client, collection=collection, vector_size=len(vector))
            point = {
                "id": item_id,
                "vector": vector,
                "payload": payload,
            }
            response = await client.put(
                f"{self.base_url}/collections/{collection}/points",
                json={"points": [point]},
            )
            response.raise_for_status()
        return None

    async def search(self, collection: str, vector: list[float], limit: int = 5) -> list[dict]:
        if not vector:
            return []

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection}/points/search",
                    json={"vector": vector, "limit": limit, "with_payload": True},
                )
                response.raise_for_status()
                body = response.json()
        except Exception:
            return []

        hits = body.get("result", [])
        if not isinstance(hits, list):
            return []
        return hits

    async def delete(self, collection: str, item_ids: list[str]) -> bool:
        if not item_ids:
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection}/points/delete",
                    json={"points": item_ids},
                )
                response.raise_for_status()
        except Exception:
            return False
        return True

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/collections")
            return response.status_code < 500
        except Exception:
            return False

    async def _ensure_collection(
        self,
        client: httpx.AsyncClient,
        collection: str,
        vector_size: int,
    ) -> None:
        if collection in self._known_collections:
            return
        response = await client.put(
            f"{self.base_url}/collections/{collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        # Qdrant may return 409 if collection already exists.
        if response.status_code not in (200, 201, 409):
            response.raise_for_status()
        self._known_collections.add(collection)
