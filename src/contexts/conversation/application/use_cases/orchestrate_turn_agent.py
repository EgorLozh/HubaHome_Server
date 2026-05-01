import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort
from src.contexts.knowledge.application.ports.vector_store_port import VectorStorePort
from src.contexts.knowledge.infrastructure.adapters.json_metadata.json_metadata_store import (
    JsonKnowledgeMetadataStore,
)
from src.contexts.webintel.application.ports.web_search_port import WebSearchPort
from src.shared.observability.metrics import ERROR_COUNT


@dataclass
class TurnOutput:
    assistant_text: str
    intent: str


class WeatherToolInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Локация для запроса погоды. Укажи город/регион и при необходимости "
            "контекст времени, например: 'Москва', 'Москва завтра утром'."
        ),
    )


class WebSearchToolInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Запрос для поиска в интернете. Используй для актуальных фактов, новостей, "
            "определений и данных, которых нет в памяти пользователя."
        ),
    )


class CreateDocumentToolInput(BaseModel):
    title: str = Field(..., description="Заголовок заметки для сохранения в памяти пользователя.")
    text: str = Field(..., description="Основной текст заметки для сохранения.")
    tags: list[str] = Field(
        default_factory=list,
        description="Необязательные теги по теме, например: ['покупки', 'рецепты'].",
    )


class UpdateDocumentToolInput(BaseModel):
    id: str = Field(..., description="ID существующей заметки, полученный при создании.")
    title: str | None = Field(default=None, description="Новый заголовок (если пользователь просит переименовать).")
    text: str | None = Field(default=None, description="Новый текст (если пользователь просит изменить содержимое).")
    tags: list[str] | None = Field(default=None, description="Новый список тегов (если пользователь явно просит).")


class DeleteDocumentToolInput(BaseModel):
    id: str = Field(..., description="ID существующей заметки для удаления из памяти.")


class SearchDocumentToolInput(BaseModel):
    query: str = Field(..., description="Запрос на естественном языке для поиска сохраненных заметок.")


class SetInstructionToolInput(BaseModel):
    key: str = Field(
        ...,
        description=(
            "Ключ постоянной инструкции. Используй стабильные ключи, например 'name' или 'tone'. "
            "Не придумывай случайные ключи, если подходит существующий."
        ),
    )
    value: str = Field(..., description="Значение инструкции для постоянного сохранения, например имя пользователя.")


class GetInstructionToolInput(BaseModel):
    key: str = Field(..., description="Ключ инструкции для чтения, например 'name' или 'tone'.")


class DeleteInstructionToolInput(BaseModel):
    key: str = Field(..., description="Ключ инструкции для удаления.")


class OrchestrateTurnUseCase:
    def __init__(
        self,
        llm_provider: LLMProviderPort,
        web_search: WebSearchPort,
        vector_store: VectorStorePort,
        knowledge_collection: str,
        metadata_store: JsonKnowledgeMetadataStore,
        max_tool_calls_per_turn: int = 3,
        max_steps_per_turn: int = 4,
        turn_timeout_ms: int = 8000,
    ) -> None:
        self.llm_provider = llm_provider
        self.web_search = web_search
        self.vector_store = vector_store
        self.knowledge_collection = knowledge_collection
        self.metadata_store = metadata_store
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_steps_per_turn = max_steps_per_turn
        self.turn_timeout_ms = turn_timeout_ms

        self._tools = self._build_tools()
        self._agent_runner = self._build_agent_runner()

    async def execute(self, transcript: str) -> TurnOutput:
        user_text = transcript.strip()
        if not user_text:
            return TurnOutput(assistant_text="Не расслышал команду. Повтори, пожалуйста.", intent="unknown")

        if self._is_reminder_intent(user_text):
            ERROR_COUNT.labels(source="unsupported_capability_reminder").inc()
            return TurnOutput(
                assistant_text=(
                    "Пока я не умею работать с напоминаниями. "
                    "Сейчас доступны погода, документы в памяти и мета-инструкции."
                ),
                intent="unsupported_capability",
            )

        try:
            prompt_context = self._build_prompt_context()
            user_prompt = (
                f"{prompt_context}\n\n"
                "Запрос пользователя:\n"
                f"{user_text}"
            )
            result = await asyncio.wait_for(
                self._agent_runner.ainvoke({"messages": [{"role": "user", "content": user_prompt}]}),
                timeout=self.turn_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            ERROR_COUNT.labels(source="agent_timeout").inc()
            return TurnOutput(
                assistant_text="Обработка запроса заняла слишком много времени. Попробуй короче сформулировать запрос.",
                intent="timeout",
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if "status code: 503" in error_text:
                ERROR_COUNT.labels(source="agent_llm_unavailable").inc()
                return TurnOutput(
                    assistant_text=(
                        "Сейчас локальная LLM временно недоступна (503). "
                        "Проверь Ollama и повтори запрос через несколько секунд."
                    ),
                    intent="error",
                )
            ERROR_COUNT.labels(source="agent_executor").inc()
            return TurnOutput(
                assistant_text="Не удалось обработать запрос. Попробуй еще раз.",
                intent="error",
            )

        messages = result.get("messages", [])
        tool_names = self._extract_tool_names(messages=messages)
        intent = self._resolve_intent(tool_names=tool_names)
        assistant_text = self._extract_assistant_text(messages=messages)
        return TurnOutput(assistant_text=assistant_text, intent=intent)

    def _build_prompt_context(self) -> str:
        instructions = self.metadata_store.get_instructions()
        context_lines = ["Ты домашний ассистент. Отвечай кратко и по-русски."]
        name = instructions.get("name")
        if name:
            context_lines.append(f"Обращайся к пользователю по имени: {name}.")
        tone = instructions.get("tone")
        if tone:
            context_lines.append(f"Стиль ответа: {tone}.")
        return "\n".join(context_lines)

    def _build_agent_runner(self):
        return create_agent(
            model=self.llm_provider.get_chat_model(),
            tools=self._tools,
            system_prompt=(
                "Ты домашний ассистент, тебя зовут Хуба. Отвечай информативно по-русски. "
                "Используй инструменты только по необходимости. "
                "Выбор инструментов: "
                "weather_tool для фактов о погоде, "
                "internet_search_tool для общего поиска в интернете по актуальным данным, "
                "create_document_tool/update_document_tool/delete_document_tool/search_documents_tool для работы с заметками пользователя (Информация, заметки, задачи, списки, etc.), "
                "set_instruction_tool/get_instruction_tool/get_instructions_tool/delete_instruction_tool для постоянных предпочтений пользователя (Правила, инструкции, предпочтения). "
                "Если у пользователя обычный вопрос без внешних данных или изменений памяти, не вызывай инструменты. "
                "Напоминания и расписание не поддерживаются: в таких запросах "
                "не вызывай инструменты и сообщай об ограничении."
            ),
        )

    def _build_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                coroutine=self._run_weather_tool,
                name="weather_tool",
                description=(
                    "Используй только для вопросов о погоде. "
                    "Вход: строка с локацией/контекстом. Выход: короткие фактические фрагменты о погоде."
                ),
                args_schema=WeatherToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._run_internet_search_tool,
                name="internet_search_tool",
                description=(
                    "Ищет актуальную информацию в интернете. "
                    "Используй для фактов и новостей вне памяти пользователя, когда вопрос не про погоду."
                ),
                args_schema=WebSearchToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._create_document_tool,
                name="create_document_tool",
                description=(
                    "Создает и сохраняет НОВУЮ заметку в памяти. "
                    "Используй, когда пользователь просит что-то запомнить или сохранить."
                ),
                args_schema=CreateDocumentToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._update_document_tool,
                name="update_document_tool",
                description=(
                    "Обновляет СУЩЕСТВУЮЩУЮ заметку по id (заголовок/текст/теги). "
                    "Не использовать для создания новой заметки. "
                    "Если id неизвестен, сначала вызови search_documents_tool и возьми id из результата."
                ),
                args_schema=UpdateDocumentToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._delete_document_tool,
                name="delete_document_tool",
                description=(
                    "Удаляет существующую сохраненную заметку по id. "
                    "Если id неизвестен, сначала вызови search_documents_tool и возьми id из результата."
                ),
                args_schema=DeleteDocumentToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._search_documents_tool,
                name="search_documents_tool",
                description=(
                    "Ищет сохраненные заметки по запросу на естественном языке. "
                    "Используй, когда пользователь просит найти или вспомнить сохраненную информацию. "
                    "Возвращает список с id, чтобы потом можно было сделать update/delete."
                ),
                args_schema=SearchDocumentToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._set_instruction_tool,
                name="set_instruction_tool",
                description=(
                    "Создает или обновляет постоянную инструкцию/предпочтение пользователя key=value, "
                    "например имя или желаемый стиль общения."
                ),
                args_schema=SetInstructionToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._get_instruction_tool,
                name="get_instruction_tool",
                description="Читает одно сохраненное предпочтение по ключу (например 'name').",
                args_schema=GetInstructionToolInput,
            ),
            StructuredTool.from_function(
                coroutine=self._get_instructions_tool,
                name="get_instructions_tool",
                description="Читает все сохраненные предпочтения пользователя (все ключи).",
            ),
            StructuredTool.from_function(
                coroutine=self._delete_instruction_tool,
                name="delete_instruction_tool",
                description="Удаляет одно сохраненное предпочтение по ключу.",
                args_schema=DeleteInstructionToolInput,
            ),
        ]

    async def _run_weather_tool(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return "Нужен город или контекст запроса для погоды."

        search_results = await self.web_search.search(f"current weather {normalized}")
        snippets = [item.get("snippet", "") for item in search_results[:3] if item.get("snippet")]
        if not snippets:
            return "Не удалось найти релевантные данные о погоде."
        return "; ".join(snippets)

    async def _run_internet_search_tool(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return "Нужен запрос для поиска в интернете."

        search_results = await self.web_search.search(normalized)
        lines: list[str] = []
        for item in search_results[:3]:
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("snippet", "")).strip()
            if not snippet:
                continue
            source = str(item.get("source", "")).strip()
            if source:
                lines.append(f"{snippet} (источник: {source})")
            else:
                lines.append(snippet)

        if not lines:
            return "Не удалось получить релевантные данные из интернета."
        return " ; ".join(lines)

    async def _create_document_tool(self, title: str, text: str, tags: list[str]) -> str:
        normalized_text = text.strip()
        normalized_title = title.strip() or "Заметка"
        if not normalized_text:
            return "Нет текста для сохранения документа."

        item_id = str(uuid4())
        vector = self._embed_text(normalized_text)
        document = self.metadata_store.create_document(
            item_id=item_id,
            title=normalized_title,
            text=normalized_text,
            tags=tags or [],
        )
        try:
            await self.vector_store.upsert(
                collection=self.knowledge_collection,
                item_id=item_id,
                vector=vector,
                payload={"title": normalized_title, "text": normalized_text},
            )
        except Exception:
            self.metadata_store.delete_document(item_id)
            return "Не удалось сохранить документ в векторную базу."
        return f"Документ сохранен: id={document.get('id')} title={document.get('title')}"

    async def _update_document_tool(
        self, id: str, title: str | None = None, text: str | None = None, tags: list[str] | None = None
    ) -> str:
        item_id = id.strip()
        if not item_id:
            return "Для обновления нужен id документа."
        existing = self.metadata_store.get_document(item_id)
        if existing is None:
            return "Документ не найден."

        updated = self.metadata_store.update_document(item_id=item_id, title=title, text=text, tags=tags)
        if updated is None:
            return "Документ не найден."
        try:
            await self.vector_store.upsert(
                collection=self.knowledge_collection,
                item_id=item_id,
                vector=self._embed_text(str(updated.get("text", ""))),
                payload={"title": updated.get("title", ""), "text": updated.get("text", "")},
            )
        except Exception:
            self.metadata_store.replace_document(item_id=item_id, document=existing)
            return "Не удалось обновить документ в векторной базе."
        return "Документ обновлен."

    async def _delete_document_tool(self, id: str) -> str:
        item_id = id.strip()
        if not item_id:
            return "Для удаления нужен id документа."
        existing = self.metadata_store.get_document(item_id)
        if existing is None:
            return "Документ не найден."

        delete_method = getattr(self.vector_store, "delete", None)
        if callable(delete_method):
            delete_result = await delete_method(self.knowledge_collection, [item_id])
            if delete_result is False:
                return "Не удалось удалить документ из векторной базы."
        existed = self.metadata_store.delete_document(item_id)
        return "Документ удален." if existed else "Документ не найден."

    async def _search_documents_tool(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return "Нужен запрос для поиска по документам."

        vector = self._embed_text(normalized)
        search_hits = await self.vector_store.search(
            collection=self.knowledge_collection,
            vector=vector,
            limit=3,
        )
        hit_ids = [str(item.get("id")) for item in search_hits if item.get("id")]
        metadata_hits = self.metadata_store.get_many(hit_ids)
        if not metadata_hits:
            metadata_hits = self.metadata_store.search(query=normalized, limit=3)
        if not metadata_hits:
            return "В памяти пока нет подходящих документов."
        summaries: list[str] = []
        for item in metadata_hits:
            item_id = str(item.get("id", "")).strip()
            title = str(item.get("title", "")).strip() or "Без названия"
            text = str(item.get("text", "")).strip()
            preview = text[:120] + ("..." if len(text) > 120 else "")
            summaries.append(f"id={item_id}; title={title}; text={preview}")
        return "Нашел документы:\n" + "\n".join(f"- {line}" for line in summaries)

    async def _set_instruction_tool(self, key: str, value: str) -> str:
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key or not normalized_value:
            return "Нужны key и value для сохранения инструкции."
        self.metadata_store.set_instruction(key=normalized_key, value=normalized_value)
        return f"Инструкция обновлена: {normalized_key}={normalized_value}"

    async def _get_instruction_tool(self, key: str) -> str:
        normalized_key = key.strip()
        if not normalized_key:
            return "Нужен ключ инструкции."
        instructions = self.metadata_store.get_instructions()
        value = instructions.get(normalized_key)
        if value is None:
            return "Инструкция не найдена."
        return f"Инструкция: {normalized_key}={value}"

    async def _get_instructions_tool(self) -> str:
        instructions = self.metadata_store.get_instructions()
        if not instructions:
            return "Постоянные инструкции пока не заданы."
        pairs = ", ".join(f"{key}={value}" for key, value in instructions.items())
        return f"Текущие инструкции: {pairs}"

    async def _delete_instruction_tool(self, key: str) -> str:
        normalized_key = key.strip()
        existed = self.metadata_store.delete_instruction(normalized_key)
        return "Инструкция удалена." if existed else "Инструкция не найдена."

    @staticmethod
    def _extract_tool_names(messages: list[Any]) -> list[str]:
        names: list[str] = []
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None)
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        return names

    @staticmethod
    def _extract_assistant_text(messages: list[Any]) -> str:
        fallback_ai_text = ""
        fallback_tool_text = ""
        for message in reversed(messages):
            role = getattr(message, "type", "")
            content_text = OrchestrateTurnUseCase._extract_text_from_content(getattr(message, "content", ""))
            if role in {"tool"} and content_text and not fallback_tool_text:
                fallback_tool_text = content_text
            if role not in {"ai", "assistant"}:
                continue
            if content_text and not fallback_ai_text:
                fallback_ai_text = content_text
            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls and not content_text:
                continue
            if content_text:
                return content_text
        if fallback_ai_text:
            return fallback_ai_text
        if fallback_tool_text:
            return f"Вот что удалось найти: {fallback_tool_text}"
        return "Не удалось подготовить ответ."

    @staticmethod
    def _resolve_intent(tool_names: list[str]) -> str:
        if not tool_names:
            return "chat"
        first_tool = tool_names[0]
        if first_tool == "weather_tool":
            return "weather"
        if first_tool == "internet_search_tool":
            return "web"
        if "document" in first_tool:
            return "knowledge"
        if "instruction" in first_tool:
            return "metadata"
        return "chat"

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            dict_text = OrchestrateTurnUseCase._extract_text_from_dict(content)
            return dict_text.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block.strip())
                    continue
                if isinstance(block, dict):
                    text_value = OrchestrateTurnUseCase._extract_text_from_dict(block)
                    if text_value:
                        text_parts.append(text_value.strip())
            return " ".join(part for part in text_parts if part).strip()

        return ""

    @staticmethod
    def _extract_text_from_dict(block: dict[str, Any]) -> str:
        for key in ("text", "content", "value"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _embed_text(text: str, dims: int = 24) -> list[float]:
        if dims <= 0:
            return []
        vector = [0.0] * dims
        normalized = text.strip().lower()
        if not normalized:
            return vector
        for index, ch in enumerate(normalized):
            slot = index % dims
            vector[slot] += (ord(ch) % 64) / 64.0
        length = max(len(normalized), 1)
        return [value / length for value in vector]

    @staticmethod
    def _is_reminder_intent(text: str) -> bool:
        lowered = text.lower()
        keywords = ("напоминан", "reminder", "расписани", "schedule")
        return any(keyword in lowered for keyword in keywords)
