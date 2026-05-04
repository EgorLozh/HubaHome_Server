import json
from datetime import datetime, timezone
from pathlib import Path


class JsonKnowledgeMetadataStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._payload_cache: dict[str, dict] | None = None
        self._payload_cache_mtime_ns: int | None = None

    def upsert(self, item_id: str, text: str) -> None:
        payload = self._read_payload()
        payload["documents"][item_id] = {
            "id": item_id,
            "title": "",
            "text": text,
            "tags": [],
            "updated_at": self._timestamp(),
            "version": 1,
        }
        self._write_payload(payload)

    def get_many(self, item_ids: list[str]) -> list[dict]:
        payload = self._read_payload()
        docs = payload["documents"]
        return [docs[item_id] for item_id in item_ids if item_id in docs]

    def search(self, query: str, limit: int = 3) -> list[dict]:
        payload = self._read_payload()
        documents = payload["documents"]
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        scored: list[tuple[int, dict]] = []
        for item in documents.values():
            text = str(item.get("text", ""))
            item_tokens = self._tokenize(text)
            overlap = len(query_tokens.intersection(item_tokens))
            scored.append((overlap, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = [item for score, item in scored if score > 0]
        if best:
            return best[:limit]
        return list(documents.values())[:limit]

    def create_document(self, item_id: str, title: str, text: str, tags: list[str] | None = None) -> dict:
        payload = self._read_payload()
        now = self._timestamp()
        document = {
            "id": item_id,
            "title": title,
            "text": text,
            "tags": tags or [],
            "updated_at": now,
            "version": 1,
        }
        payload["documents"][item_id] = document
        self._write_payload(payload)
        return document

    def update_document(
        self,
        item_id: str,
        title: str | None = None,
        text: str | None = None,
        tags: list[str] | None = None,
    ) -> dict | None:
        payload = self._read_payload()
        documents = payload["documents"]
        existing = documents.get(item_id)
        if existing is None:
            return None

        updated = dict(existing)
        if title is not None:
            updated["title"] = title
        if text is not None:
            updated["text"] = text
        if tags is not None:
            updated["tags"] = tags
        updated["version"] = int(existing.get("version", 0)) + 1
        updated["updated_at"] = self._timestamp()
        documents[item_id] = updated
        self._write_payload(payload)
        return updated

    def get_document(self, item_id: str) -> dict | None:
        payload = self._read_payload()
        document = payload["documents"].get(item_id)
        if document is None:
            return None
        return dict(document)

    def replace_document(self, item_id: str, document: dict) -> None:
        payload = self._read_payload()
        payload["documents"][item_id] = dict(document)
        self._write_payload(payload)

    def delete_document(self, item_id: str) -> bool:
        payload = self._read_payload()
        documents = payload["documents"]
        existed = item_id in documents
        if existed:
            del documents[item_id]
            self._write_payload(payload)
        return existed

    def set_instruction(self, key: str, value: str) -> None:
        payload = self._read_payload()
        payload["instructions"][key] = value
        self._write_payload(payload)

    def delete_instruction(self, key: str) -> bool:
        payload = self._read_payload()
        instructions = payload["instructions"]
        existed = key in instructions
        if existed:
            del instructions[key]
            self._write_payload(payload)
        return existed

    def get_instructions(self) -> dict[str, str]:
        payload = self._read_payload()
        return dict(payload["instructions"])

    def _read_payload(self) -> dict[str, dict]:
        if not self._path.exists():
            payload = {"instructions": {}, "documents": {}}
            self._payload_cache = self._clone_payload(payload)
            self._payload_cache_mtime_ns = None
            return payload
        current_mtime_ns = self._path.stat().st_mtime_ns
        if self._payload_cache is not None and self._payload_cache_mtime_ns == current_mtime_ns:
            return self._clone_payload(self._payload_cache)
        try:
            content = self._path.read_text(encoding="utf-8").strip()
            if not content:
                payload = {"instructions": {}, "documents": {}}
                self._payload_cache = self._clone_payload(payload)
                self._payload_cache_mtime_ns = current_mtime_ns
                return payload
            data = json.loads(content)
            if isinstance(data, dict):
                if "instructions" in data or "documents" in data:
                    payload = {
                        "instructions": dict(data.get("instructions", {})),
                        "documents": dict(data.get("documents", {})),
                    }
                    self._payload_cache = self._clone_payload(payload)
                    self._payload_cache_mtime_ns = current_mtime_ns
                    return payload
                # Backward compatibility with legacy flat doc map.
                payload = {"instructions": {}, "documents": dict(data)}
                self._payload_cache = self._clone_payload(payload)
                self._payload_cache_mtime_ns = current_mtime_ns
                return payload
        except Exception:
            payload = {"instructions": {}, "documents": {}}
            self._payload_cache = self._clone_payload(payload)
            self._payload_cache_mtime_ns = current_mtime_ns
            return payload
        payload = {"instructions": {}, "documents": {}}
        self._payload_cache = self._clone_payload(payload)
        self._payload_cache_mtime_ns = current_mtime_ns
        return payload

    def _write_payload(self, payload: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._payload_cache = self._clone_payload(payload)
        self._payload_cache_mtime_ns = self._path.stat().st_mtime_ns

    @staticmethod
    def _clone_payload(payload: dict[str, dict]) -> dict[str, dict]:
        return {
            "instructions": dict(payload.get("instructions", {})),
            "documents": {
                str(item_id): dict(document) if isinstance(document, dict) else document
                for item_id, document in payload.get("documents", {}).items()
            },
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        sanitized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {token for token in sanitized.split() if token}

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
