import asyncio
import re
import tempfile
from dataclasses import dataclass
from typing import Any

from src.contexts.conversation.application.use_cases.orchestrate_turn_agent import OrchestrateTurnUseCase
from src.contexts.knowledge.infrastructure.adapters.json_metadata.json_metadata_store import (
    JsonKnowledgeMetadataStore,
)


class FakeLlm:
    def get_chat_model(self):
        raise RuntimeError("not used in tests")

    async def generate_text(self, prompt: str) -> str:
        return "ok"

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


@dataclass
class FakeMessage:
    type: str
    content: Any
    tool_calls: list[dict] | None = None


class FakeExecutor:
    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.last_config: dict[str, Any] | None = None

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.last_config = config
        return self._response_factory(payload)


class UseCaseForTests(OrchestrateTurnUseCase):
    def __init__(self, fake_executor: FakeExecutor, **kwargs: Any) -> None:
        self._fake_executor = fake_executor
        super().__init__(**kwargs)

    def _build_agent_runner(self):
        return self._fake_executor


def _build_use_case(response_factory) -> UseCaseForTests:
    temp_dir = tempfile.mkdtemp()
    metadata_store = JsonKnowledgeMetadataStore(file_path=f"{temp_dir}/knowledge.json")
    return UseCaseForTests(
        fake_executor=FakeExecutor(response_factory=response_factory),
        llm_provider=FakeLlm(),
        web_search=FakeWebSearch(),
        vector_store=FakeVectorStore(),
        knowledge_collection="test_notes",
        metadata_store=metadata_store,
        max_tool_calls_per_turn=3,
        max_steps_per_turn=4,
    )


def test_orchestrate_turn_weather_intent():
    use_case = _build_use_case(
        lambda _: {
            "messages": [
                FakeMessage(type="ai", content="", tool_calls=[{"name": "weather_tool"}]),
                FakeMessage(type="ai", content="Сегодня около +12, облачно.", tool_calls=[]),
            ],
        }
    )
    result = asyncio.run(use_case.execute("какая погода в москве"))

    assert result.intent == "weather"
    assert "облачно" in result.assistant_text


def test_orchestrate_turn_reminder_intent_is_out_of_scope():
    use_case = _build_use_case(lambda _: {"messages": []})
    result = asyncio.run(use_case.execute("поставь напоминание на 9 утра"))

    assert result.intent == "unsupported_capability"
    assert "напоминаниями" in result.assistant_text


def test_orchestrate_turn_chat_without_tools():
    use_case = _build_use_case(
        lambda _: {
            "messages": [
                FakeMessage(type="ai", content="Привет!", tool_calls=[]),
            ],
        }
    )
    result = asyncio.run(use_case.execute("привет"))

    assert result.intent == "chat"
    assert result.assistant_text == "Привет!"


def test_orchestrate_turn_web_search_intent():
    use_case = _build_use_case(
        lambda _: {
            "messages": [
                FakeMessage(type="ai", content="", tool_calls=[{"name": "internet_search_tool"}]),
                FakeMessage(type="ai", content="Нашел актуальные новости в интернете.", tool_calls=[]),
            ],
        }
    )
    result = asyncio.run(use_case.execute("что нового в мире технологий"))

    assert result.intent == "web"
    assert "новости" in result.assistant_text


def test_orchestrate_turn_weather_uses_tool_fallback_when_final_ai_text_missing():
    use_case = _build_use_case(
        lambda _: {
            "messages": [
                FakeMessage(type="ai", content="", tool_calls=[{"name": "weather_tool"}]),
                FakeMessage(type="tool", content="В Ижевске около +12, облачно."),
            ],
        }
    )
    result = asyncio.run(use_case.execute("какая сейчас погода в ижевске"))

    assert result.intent == "weather"
    assert "Ижевске" in result.assistant_text
    assert result.assistant_text != "Не удалось подготовить ответ."


def test_orchestrate_turn_applies_metadata_instruction_to_prompt():
    def response_factory(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        first_content = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                first_content = str(first.get("content", ""))
        context = first_content
        if "Обращайся к пользователю по имени: Егор." in context:
            return {"messages": [FakeMessage(type="ai", content="Егор, рад помочь.", tool_calls=[])]}
        return {"messages": [FakeMessage(type="ai", content="Рад помочь.", tool_calls=[])]}

    use_case = _build_use_case(response_factory)
    asyncio.run(use_case._set_instruction_tool(key="name", value="Егор"))
    response = asyncio.run(use_case.execute("привет"))

    assert response.intent == "chat"
    assert "Егор" in response.assistant_text


def test_orchestrate_turn_includes_all_persistent_instructions_in_prompt():
    def response_factory(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        first_content = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                first_content = str(first.get("content", ""))
        if "- preferred_answer_style: сначала короткий вывод, потом детали" in first_content:
            return {"messages": [FakeMessage(type="ai", content="Учел постоянные инструкции.", tool_calls=[])]}
        return {"messages": [FakeMessage(type="ai", content="Инструкции не найдены.", tool_calls=[])]}

    use_case = _build_use_case(response_factory)
    asyncio.run(use_case._set_instruction_tool("preferred_answer_style", "сначала короткий вывод, потом детали"))
    response = asyncio.run(use_case.execute("объясни статус сервера"))

    assert response.intent == "chat"
    assert response.assistant_text == "Учел постоянные инструкции."


def test_orchestrate_turn_passes_recursion_limit_to_agent_runner():
    use_case = _build_use_case(lambda _: {"messages": [FakeMessage(type="ai", content="ok", tool_calls=[])]})
    response = asyncio.run(use_case.execute("привет"))

    assert response.assistant_text == "ok"
    assert use_case._fake_executor.last_config == {"recursion_limit": 9}


def test_knowledge_document_update_and_delete_via_tools():
    use_case = _build_use_case(lambda _: {"messages": [FakeMessage(type="ai", content="ok", tool_calls=[])]})
    created = asyncio.run(
        use_case._create_document_tool(
            title="T1",
            text="старый текст",
            tags=[],
        )
    )
    match = re.search(r"id=([a-f0-9-]+)", created)
    assert match is not None
    item_id = match.group(1)

    updated = asyncio.run(
        use_case._update_document_tool(
            id=item_id,
            text="новый текст",
        )
    )
    searched = asyncio.run(
        use_case._search_documents_tool(
            query="новый",
        )
    )
    deleted = asyncio.run(
        use_case._delete_document_tool(
            id=item_id,
        )
    )

    assert "сохранен" in created
    assert updated == "Документ обновлен."
    assert "новый текст" in searched
    assert deleted == "Документ удален."


def test_knowledge_document_update_rolls_back_on_vector_failure():
    use_case = _build_use_case(lambda _: {"messages": [FakeMessage(type="ai", content="ok", tool_calls=[])]})
    vector_store: Any = use_case.vector_store
    created = asyncio.run(
        use_case._create_document_tool(
            title="T2",
            text="до сбоя",
            tags=[],
        )
    )
    match = re.search(r"id=([a-f0-9-]+)", created)
    assert match is not None
    item_id = match.group(1)

    vector_store.raise_on_upsert = True
    failed_update = asyncio.run(
        use_case._update_document_tool(
            id=item_id,
            text="после сбоя",
        )
    )
    stored_document = use_case.metadata_store.get_document(item_id)

    assert failed_update == "Не удалось обновить документ в векторной базе."
    assert stored_document is not None
    assert stored_document.get("text") == "до сбоя"
