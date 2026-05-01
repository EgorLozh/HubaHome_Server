import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
from uuid import uuid4

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort
from src.contexts.agent.application.tools.tool_registry import ToolRegistry
from src.contexts.agent.application.tools.tool_spec import ToolCall, ToolResult, ToolSpec
from src.contexts.knowledge.application.ports.vector_store_port import VectorStorePort
from src.contexts.knowledge.infrastructure.adapters.json_metadata.json_metadata_store import (
    JsonKnowledgeMetadataStore,
)
from src.contexts.notification.application.ports.notification_port import NotificationPort
from src.contexts.webintel.application.ports.web_search_port import WebSearchPort
from src.shared.observability.metrics import ERROR_COUNT

try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - runtime fallback
    END = "end"
    START = "start"
    HAS_LANGGRAPH = False


@dataclass
class TurnOutput:
    assistant_text: str
    intent: str


class AgentState(TypedDict, total=False):
    user_text: str
    prompt_context: str
    tool_plan: list[ToolCall]
    tool_results: list[ToolResult]
    assistant_text: str
    intent: str
    steps: int
    errors: list[str]


class OrchestrateTurnUseCase:
    def __init__(
        self,
        llm_provider: LLMProviderPort,
        web_search: WebSearchPort,
        vector_store: VectorStorePort,
        notification: NotificationPort,
        knowledge_collection: str,
        metadata_store: JsonKnowledgeMetadataStore,
        max_tool_calls_per_turn: int = 3,
        max_steps_per_turn: int = 4,
        turn_timeout_ms: int = 8000,
    ) -> None:
        self.llm_provider = llm_provider
        self.web_search = web_search
        self.vector_store = vector_store
        self.notification = notification
        self.knowledge_collection = knowledge_collection
        self.metadata_store = metadata_store
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_steps_per_turn = max_steps_per_turn
        self.turn_timeout_ms = turn_timeout_ms
        self.tool_registry = ToolRegistry(specs=self._build_tool_specs())
        self._graph = self._build_graph() if HAS_LANGGRAPH else None

    async def execute(self, transcript: str) -> TurnOutput:
        user_text = transcript.strip()
        if not user_text:
            return TurnOutput(assistant_text="Не расслышал команду. Повтори, пожалуйста.", intent="unknown")

        initial_state: AgentState = {
            "user_text": user_text,
            "tool_plan": [],
            "tool_results": [],
            "steps": 0,
            "errors": [],
        }
        try:
            if self._graph is not None:
                final_state = await asyncio.wait_for(
                    self._graph.ainvoke(initial_state), timeout=self.turn_timeout_ms / 1000
                )
            else:
                final_state = await asyncio.wait_for(
                    self._run_graph_fallback(initial_state), timeout=self.turn_timeout_ms / 1000
                )
        except asyncio.TimeoutError:
            ERROR_COUNT.labels(source="agent_timeout").inc()
            return TurnOutput(
                assistant_text="Обработка запроса заняла слишком много времени. Попробуй короче сформулировать запрос.",
                intent="timeout",
            )
        except Exception:
            ERROR_COUNT.labels(source="agent_graph").inc()
            return TurnOutput(
                assistant_text="Не удалось обработать запрос. Попробуй еще раз.",
                intent="error",
            )

        return TurnOutput(
            assistant_text=str(final_state.get("assistant_text", "")).strip()
            or "Не удалось подготовить ответ.",
            intent=str(final_state.get("intent", "chat")),
        )

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("load_metadata_context", self._load_metadata_context_node)
        graph_builder.add_node("plan_tools_node", self._plan_tools_node)
        graph_builder.add_node("execute_tools_node", self._execute_tools_node)
        graph_builder.add_node("compose_answer_node", self._compose_answer_node)

        graph_builder.add_edge(START, "load_metadata_context")
        graph_builder.add_edge("load_metadata_context", "plan_tools_node")
        graph_builder.add_conditional_edges(
            "plan_tools_node",
            self._route_after_plan,
            {
                "execute_tools_node": "execute_tools_node",
                "compose_answer_node": "compose_answer_node",
            },
        )
        graph_builder.add_edge("execute_tools_node", "compose_answer_node")
        graph_builder.add_edge("compose_answer_node", END)
        return graph_builder.compile()

    async def _run_graph_fallback(self, state: AgentState) -> AgentState:
        state = self._merge_state(state, await self._load_metadata_context_node(state))
        state = self._merge_state(state, await self._plan_tools_node(state))
        route = self._route_after_plan(state)
        if route == "execute_tools_node":
            state = self._merge_state(state, await self._execute_tools_node(state))
        state = self._merge_state(state, await self._compose_answer_node(state))
        return state

    async def _load_metadata_context_node(self, state: AgentState) -> AgentState:
        instructions = self.metadata_store.get_instructions()
        context_lines = ["Ты домашний ассистент. Отвечай кратко и по-русски."]
        name = instructions.get("name")
        if name:
            context_lines.append(f"Обращайся к пользователю по имени: {name}.")
        tone = instructions.get("tone")
        if tone:
            context_lines.append(f"Стиль ответа: {tone}.")
        return {
            "prompt_context": "\n".join(context_lines),
            "steps": int(state.get("steps", 0)) + 1,
        }

    async def _plan_tools_node(self, state: AgentState) -> AgentState:
        user_text = str(state.get("user_text", ""))
        if self._is_reminder_intent(user_text):
            ERROR_COUNT.labels(source="unsupported_capability_reminder").inc()
            return {
                "assistant_text": (
                    "Пока я не умею работать с напоминаниями. "
                    "Сейчас доступны погода, документы в памяти и мета-инструкции."
                ),
                "intent": "unsupported_capability",
                "tool_plan": [],
                "steps": int(state.get("steps", 0)) + 1,
            }

        planned_calls = await self._plan_tools(
            user_text=user_text,
            prompt_context=str(state.get("prompt_context", "")),
        )
        return {"tool_plan": planned_calls, "steps": int(state.get("steps", 0)) + 1}

    def _route_after_plan(self, state: AgentState) -> Literal["execute_tools_node", "compose_answer_node"]:
        if state.get("assistant_text"):
            return "compose_answer_node"
        tool_plan = state.get("tool_plan", [])
        if not tool_plan:
            return "compose_answer_node"
        if int(state.get("steps", 0)) >= self.max_steps_per_turn:
            ERROR_COUNT.labels(source="agent_budget_steps").inc()
            return "compose_answer_node"
        return "execute_tools_node"

    async def _execute_tools_node(self, state: AgentState) -> AgentState:
        tool_plan = list(state.get("tool_plan", []))[: self.max_tool_calls_per_turn]
        tool_results: list[ToolResult] = []
        for tool_call in tool_plan:
            result = await self.tool_registry.call(name=tool_call.name, payload=tool_call.input)
            tool_results.append(result)
        return {
            "tool_results": tool_results,
            "steps": int(state.get("steps", 0)) + 1,
        }

    async def _compose_answer_node(self, state: AgentState) -> AgentState:
        if state.get("assistant_text"):
            return {
                "assistant_text": str(state.get("assistant_text", "")),
                "intent": str(state.get("intent", "chat")),
            }

        user_text = str(state.get("user_text", ""))
        prompt_context = str(state.get("prompt_context", ""))
        tool_results = state.get("tool_results", [])
        if tool_results:
            assistant_text = await self._compose_answer(
                user_text=user_text,
                prompt_context=prompt_context,
                tool_results=tool_results,
            )
            intent = self._resolve_intent(tool_results=tool_results)
            return {
                "assistant_text": assistant_text,
                "intent": intent,
            }

        answer = await self.llm_provider.generate_text(
            f"{prompt_context}\n\nЗапрос пользователя: {user_text}"
        )
        return {"assistant_text": answer.strip(), "intent": "chat"}

    async def _plan_tools(self, user_text: str, prompt_context: str) -> list[ToolCall]:
        planner_prompt = (
            f"{prompt_context}\n\n"
            "Ты planner для tool-calling ассистента. Доступные инструменты:\n"
            "1) weather_tool -> input: {query}\n"
            "2) knowledge_document_tool -> input: {action, id?, title?, text?, query?, tags?}\n"
            "   action in [create_document, update_document, delete_document, search_documents]\n"
            "3) metadata_instruction_tool -> input: {action, key?, value?}\n"
            "   action in [set_instruction, update_instruction, get_instruction, get_instructions, delete_instruction]\n\n"
            "Верни только JSON формата:\n"
            '{"tools":[{"name":"weather_tool","input":{"query":"..."}}]}\n'
            "Если инструменты не нужны, верни {\"tools\":[]}.\n"
            "Если запрос про напоминания/расписание, тоже верни пустой список.\n"
            f"Запрос пользователя: {user_text}"
        )
        planner_answer = await self.llm_provider.generate_text(planner_prompt)
        planned_calls = self._parse_tool_calls(planner_answer)
        if planned_calls:
            return planned_calls
        return self._fallback_plan(user_text)

    async def _compose_answer(
        self, user_text: str, prompt_context: str, tool_results: list[ToolResult]
    ) -> str:
        lines: list[str] = []
        for result in tool_results:
            status = "ok" if result.ok else "error"
            lines.append(f"- {result.tool_name} ({status}): {result.message}")
        prompt = (
            f"{prompt_context}\n"
            "Сформируй короткий и вежливый ответ на русском.\n"
            f"Запрос пользователя: {user_text}\n"
            "Результаты инструментов:\n"
            + "\n".join(lines)
        )
        answer = await self.llm_provider.generate_text(prompt)
        if answer.strip():
            return answer.strip()
        return "\n".join(lines)

    def _build_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="weather_tool",
                description="Get weather facts via web search",
                tool_kind="read",
                requires_confirmation=False,
                execute=self._run_weather_tool,
            ),
            ToolSpec(
                name="knowledge_document_tool",
                description="Create/update/delete/search user documents",
                tool_kind="write",
                requires_confirmation=False,
                execute=self._run_knowledge_document_tool,
            ),
            ToolSpec(
                name="metadata_instruction_tool",
                description="Set/read/delete persistent instruction metadata",
                tool_kind="write",
                requires_confirmation=False,
                execute=self._run_metadata_instruction_tool,
            ),
        ]

    async def _run_weather_tool(self, payload: dict) -> ToolResult:
        query = str(payload.get("query") or "").strip()
        if not query:
            return ToolResult(
                tool_name="weather_tool",
                ok=False,
                message="Нужен город или контекст запроса для погоды.",
                data={},
            )

        search_results = await self.web_search.search(f"current weather {query}")
        snippets = [item.get("snippet", "") for item in search_results[:3] if item.get("snippet")]
        if not snippets:
            return ToolResult(
                tool_name="weather_tool",
                ok=False,
                message="Не удалось найти релевантные данные о погоде.",
                data={},
            )
        return ToolResult(
            tool_name="weather_tool",
            ok=True,
            message="; ".join(snippets),
            data={"items": search_results[:3]},
        )

    async def _run_knowledge_document_tool(self, payload: dict) -> ToolResult:
        action = str(payload.get("action") or "search_documents").strip().lower()
        if action in {"save", "create_document"}:
            text = str(payload.get("text") or "").strip()
            title = str(payload.get("title") or "Заметка").strip()
            if not text:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Нет текста для сохранения документа.",
                    data={},
                )
            item_id = str(uuid4())
            vector = self._embed_text(text)
            document = self.metadata_store.create_document(item_id=item_id, title=title, text=text, tags=[])
            try:
                await self.vector_store.upsert(
                    collection=self.knowledge_collection,
                    item_id=item_id,
                    vector=vector,
                    payload={"title": title, "text": text},
                )
            except Exception:
                self.metadata_store.delete_document(item_id)
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Не удалось сохранить документ в векторную базу.",
                    data={},
                )
            return ToolResult(
                tool_name="knowledge_document_tool",
                ok=True,
                message=f"Документ сохранен: {title}",
                data={"document": document},
            )

        if action == "update_document":
            item_id = str(payload.get("id") or "").strip()
            if not item_id:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Для обновления нужен id документа.",
                    data={},
                )
            existing = self.metadata_store.get_document(item_id)
            if existing is None:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Документ не найден.",
                    data={},
                )
            updated = self.metadata_store.update_document(
                item_id=item_id,
                title=payload.get("title"),
                text=payload.get("text"),
                tags=payload.get("tags"),
            )
            if updated is None:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Документ не найден.",
                    data={},
                )
            try:
                await self.vector_store.upsert(
                    collection=self.knowledge_collection,
                    item_id=item_id,
                    vector=self._embed_text(str(updated.get("text", ""))),
                    payload={"title": updated.get("title", ""), "text": updated.get("text", "")},
                )
            except Exception:
                self.metadata_store.replace_document(item_id=item_id, document=existing)
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Не удалось обновить документ в векторной базе.",
                    data={"id": item_id},
                )
            return ToolResult(
                tool_name="knowledge_document_tool",
                ok=True,
                message="Документ обновлен.",
                data={"document": updated},
            )

        if action == "delete_document":
            item_id = str(payload.get("id") or "").strip()
            if not item_id:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Для удаления нужен id документа.",
                    data={},
                )
            existing = self.metadata_store.get_document(item_id)
            if existing is None:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Документ не найден.",
                    data={"id": item_id},
                )

            delete_ok = True
            delete_method = getattr(self.vector_store, "delete", None)
            if callable(delete_method):
                delete_result = await delete_method(self.knowledge_collection, [item_id])
                if delete_result is False:
                    delete_ok = False
            if not delete_ok:
                return ToolResult(
                    tool_name="knowledge_document_tool",
                    ok=False,
                    message="Не удалось удалить документ из векторной базы.",
                    data={"id": item_id},
                )
            existed = self.metadata_store.delete_document(item_id)
            return ToolResult(
                tool_name="knowledge_document_tool",
                ok=existed,
                message="Документ удален." if existed else "Документ не найден.",
                data={"id": item_id},
            )

        query = str(payload.get("query") or "").strip()
        if not query:
            return ToolResult(
                tool_name="knowledge_document_tool",
                ok=False,
                message="Нужен запрос для поиска по документам.",
                data={},
            )
        vector = self._embed_text(query)
        search_hits = await self.vector_store.search(
            collection=self.knowledge_collection,
            vector=vector,
            limit=3,
        )
        hit_ids = [str(item.get("id")) for item in search_hits if item.get("id")]
        metadata_hits = self.metadata_store.get_many(hit_ids)
        if not metadata_hits:
            metadata_hits = self.metadata_store.search(query=query, limit=3)
        if not metadata_hits:
            return ToolResult(
                tool_name="knowledge_document_tool",
                ok=True,
                message="В памяти пока нет подходящих документов.",
                data={"items": []},
            )
        snippets = "; ".join(item.get("text", "") for item in metadata_hits if item.get("text"))
        return ToolResult(
            tool_name="knowledge_document_tool",
            ok=True,
            message=f"Нашел документы: {snippets}",
            data={"items": metadata_hits},
        )

    async def _run_metadata_instruction_tool(self, payload: dict) -> ToolResult:
        action = str(payload.get("action") or "get_instructions").strip().lower()
        if action in {"set_instruction", "update_instruction"}:
            key = str(payload.get("key") or "").strip()
            value = str(payload.get("value") or "").strip()
            if not key or not value:
                return ToolResult(
                    tool_name="metadata_instruction_tool",
                    ok=False,
                    message="Нужны key и value для сохранения инструкции.",
                    data={},
                )
            self.metadata_store.set_instruction(key=key, value=value)
            return ToolResult(
                tool_name="metadata_instruction_tool",
                ok=True,
                message=f"Инструкция обновлена: {key}={value}",
                data={"key": key, "value": value},
            )

        if action == "delete_instruction":
            key = str(payload.get("key") or "").strip()
            existed = self.metadata_store.delete_instruction(key)
            return ToolResult(
                tool_name="metadata_instruction_tool",
                ok=existed,
                message="Инструкция удалена." if existed else "Инструкция не найдена.",
                data={"key": key},
            )

        if action == "get_instruction":
            key = str(payload.get("key") or "").strip()
            instructions = self.metadata_store.get_instructions()
            if not key:
                return ToolResult(
                    tool_name="metadata_instruction_tool",
                    ok=False,
                    message="Нужен ключ инструкции.",
                    data={"instructions": instructions},
                )
            value = instructions.get(key)
            if value is None:
                return ToolResult(
                    tool_name="metadata_instruction_tool",
                    ok=False,
                    message="Инструкция не найдена.",
                    data={"key": key},
                )
            return ToolResult(
                tool_name="metadata_instruction_tool",
                ok=True,
                message=f"Инструкция: {key}={value}",
                data={"key": key, "value": value},
            )

        instructions = self.metadata_store.get_instructions()
        if not instructions:
            return ToolResult(
                tool_name="metadata_instruction_tool",
                ok=True,
                message="Постоянные инструкции пока не заданы.",
                data={"instructions": {}},
            )
        pairs = ", ".join(f"{key}={value}" for key, value in instructions.items())
        return ToolResult(
            tool_name="metadata_instruction_tool",
            ok=True,
            message=f"Текущие инструкции: {pairs}",
            data={"instructions": instructions},
        )

    def _fallback_plan(self, user_text: str) -> list[ToolCall]:
        lowered = user_text.lower()
        if self._is_weather_intent(lowered):
            return [ToolCall(name="weather_tool", input={"query": user_text})]
        if self._is_metadata_instruction_intent(lowered):
            return [
                ToolCall(
                    name="metadata_instruction_tool",
                    input={
                        "action": "set_instruction",
                        "key": "name",
                        "value": self._extract_name(user_text),
                    },
                )
            ]
        if self._is_metadata_instruction_read_intent(lowered):
            return [
                ToolCall(
                    name="metadata_instruction_tool",
                    input={"action": "get_instruction", "key": "name"},
                )
            ]
        if self._is_knowledge_save_intent(lowered):
            return [ToolCall(name="knowledge_document_tool", input={"action": "create_document", "title": "Заметка", "text": user_text})]
        if self._is_knowledge_retrieve_intent(lowered):
            return [ToolCall(name="knowledge_document_tool", input={"action": "search_documents", "query": user_text})]
        return []

    def _resolve_intent(self, tool_results: list[ToolResult]) -> str:
        if not tool_results:
            return "chat"
        tool_name = tool_results[0].tool_name
        if tool_name == "knowledge_document_tool":
            return "knowledge"
        if tool_name == "metadata_instruction_tool":
            return "metadata"
        if tool_name == "weather_tool":
            return "weather"
        return "chat"

    @staticmethod
    def _parse_tool_calls(raw_text: str) -> list[ToolCall]:
        maybe_json = raw_text.strip()
        if not maybe_json:
            return []
        try:
            payload = json.loads(maybe_json)
        except json.JSONDecodeError:
            return []
        tools = payload.get("tools", [])
        if not isinstance(tools, list):
            return []
        parsed: list[ToolCall] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            tool_input = item.get("input", {})
            if not isinstance(name, str) or not isinstance(tool_input, dict):
                continue
            if name in {"weather_tool", "knowledge_document_tool", "metadata_instruction_tool"}:
                parsed.append(ToolCall(name=name, input=tool_input))
        return parsed

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
    def _merge_state(base: AgentState, patch: AgentState) -> AgentState:
        merged = dict(base)
        merged.update(patch)
        return merged

    @staticmethod
    def _extract_name(text: str) -> str:
        lowered = text.lower()
        markers = ("меня зовут ", "my name is ")
        for marker in markers:
            if marker in lowered:
                index = lowered.index(marker) + len(marker)
                return text[index:].strip() or "друг"
        return "друг"

    @staticmethod
    def _is_weather_intent(text: str) -> bool:
        lowered = text.lower()
        keywords = ("погод", "weather", "температур", "дожд", "снег", "ветер")
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _is_knowledge_save_intent(text: str) -> bool:
        keywords = ("запомни", "сохрани", "remember this", "save this")
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _is_knowledge_retrieve_intent(text: str) -> bool:
        keywords = ("что ты знаешь", "найди документ", "remembered", "find note")
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _is_metadata_instruction_intent(text: str) -> bool:
        keywords = ("меня зовут", "обращайся ко мне", "my name is", "call me")
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _is_metadata_instruction_read_intent(text: str) -> bool:
        keywords = ("как меня зовут", "как ты меня называешь", "what is my name")
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _is_reminder_intent(text: str) -> bool:
        lowered = text.lower()
        keywords = ("напоминан", "reminder", "расписани", "schedule")
        return any(keyword in lowered for keyword in keywords)
