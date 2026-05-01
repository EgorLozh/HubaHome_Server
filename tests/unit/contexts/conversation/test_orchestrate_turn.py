import asyncio
import tempfile
from typing import Any

from src.contexts.conversation.application.use_cases.orchestrate_turn import OrchestrateTurnUseCase
from src.contexts.knowledge.infrastructure.adapters.json_metadata.json_metadata_store import (
    JsonKnowledgeMetadataStore,
)


class FakeLlm:
    async def generate_text(self, prompt: str) -> str:
        if '"tools"' in prompt:
            if "погода" in prompt.lower():
                return '{"tools":[{"name":"weather_tool","input":{"query":"москва"}}]}'
            if "запомни" in prompt.lower():
                return (
                    '{"tools":[{"name":"knowledge_document_tool","input":'
                    '{"action":"create_document","title":"Заметка","text":"моя заметка"}}]}'
                )
            if "что ты знаешь" in prompt.lower():
                return (
                    '{"tools":[{"name":"knowledge_document_tool","input":'
                    '{"action":"search_documents","query":"заметка"}}]}'
                )
            if "меня зовут" in prompt.lower():
                return (
                    '{"tools":[{"name":"metadata_instruction_tool","input":'
                    '{"action":"set_instruction","key":"name","value":"Егор"}}]}'
                )
            if "как меня зовут" in prompt.lower():
                return (
                    '{"tools":[{"name":"metadata_instruction_tool","input":'
                    '{"action":"get_instruction","key":"name"}}]}'
                )
            return '{"tools":[]}'
        if "Обращайся к пользователю по имени: Егор." in prompt:
            return "Егор, сегодня около +12, облачно."
        return "Сегодня около +12, облачно."

    async def ping(self) -> bool:
        return True


class FakeWebSearch:
    async def search(self, query: str) -> list[dict]:
        return [{"source": "test", "snippet": f"Weather data for query: {query}"}]


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.raise_on_upsert = False
        self.raise_on_delete = False

    async def upsert(self, collection: str, item_id: str, vector: list[float], payload: dict) -> None:
        if self.raise_on_upsert:
            raise RuntimeError("upsert failed")
        self.items[item_id] = {"id": item_id, "payload": payload}

    async def search(self, collection: str, vector: list[float], limit: int = 5) -> list[dict]:
        return [{"id": item_id} for item_id in list(self.items.keys())[:limit]]

    async def delete(self, collection: str, item_ids: list[str]) -> bool:
        if self.raise_on_delete:
            return False
        for item_id in item_ids:
            self.items.pop(item_id, None)
        return True

    async def ping(self) -> bool:
        return True


class FakeNotification:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str, reminder_id: str | None = None, scheduled_at: str | None = None) -> None:
        self.sent.append(text)


def _build_use_case() -> OrchestrateTurnUseCase:
    temp_dir = tempfile.mkdtemp()
    metadata_store = JsonKnowledgeMetadataStore(file_path=f"{temp_dir}/knowledge.json")
    return OrchestrateTurnUseCase(
        llm_provider=FakeLlm(),
        web_search=FakeWebSearch(),
        vector_store=FakeVectorStore(),
        notification=FakeNotification(),
        knowledge_collection="test_notes",
        metadata_store=metadata_store,
        max_tool_calls_per_turn=3,
    )


def test_orchestrate_turn_weather_tool():
    use_case = _build_use_case()
    result = asyncio.run(use_case.execute("какая погода в москве"))

    assert result.intent == "weather"
    assert "облачно" in result.assistant_text


def test_orchestrate_turn_reminder_intent_is_out_of_scope():
    use_case = _build_use_case()
    result = asyncio.run(use_case.execute("поставь напоминание на 9 утра"))

    assert result.intent == "unsupported_capability"
    assert "напоминаниями" in result.assistant_text


def test_orchestrate_turn_knowledge_save_and_retrieve():
    use_case = _build_use_case()
    saved = asyncio.run(use_case.execute("запомни моя заметка"))
    found = asyncio.run(use_case.execute("что ты знаешь про заметка"))

    assert saved.intent == "knowledge"
    assert found.intent == "knowledge"


def test_orchestrate_turn_applies_metadata_instruction_to_prompt():
    use_case = _build_use_case()
    set_name = asyncio.run(use_case.execute("меня зовут Егор"))
    response = asyncio.run(use_case.execute("привет"))

    assert set_name.intent == "metadata"
    assert response.intent == "chat"
    assert "Егор" in response.assistant_text


def test_knowledge_document_update_and_delete_via_tools():
    use_case = _build_use_case()
    created = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "create_document", "title": "T1", "text": "старый текст"},
        )
    )
    document = created.data["document"]
    item_id = str(document["id"])

    updated = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "update_document", "id": item_id, "text": "новый текст"},
        )
    )
    searched = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "search_documents", "query": "новый"},
        )
    )
    deleted = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "delete_document", "id": item_id},
        )
    )

    assert created.ok is True
    assert updated.ok is True
    assert searched.ok is True
    assert "новый текст" in searched.message
    assert deleted.ok is True


def test_knowledge_document_update_rolls_back_on_vector_failure():
    use_case = _build_use_case()
    vector_store: Any = use_case.vector_store
    created = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "create_document", "title": "T2", "text": "до сбоя"},
        )
    )
    item_id = str(created.data["document"]["id"])
    # Simulate vector layer failure during update; metadata should rollback.
    vector_store.raise_on_upsert = True
    failed_update = asyncio.run(
        use_case.tool_registry.call(
            "knowledge_document_tool",
            {"action": "update_document", "id": item_id, "text": "после сбоя"},
        )
    )
    stored_document = use_case.metadata_store.get_document(item_id)

    assert failed_update.ok is False
    assert stored_document is not None
    assert stored_document.get("text") == "до сбоя"
