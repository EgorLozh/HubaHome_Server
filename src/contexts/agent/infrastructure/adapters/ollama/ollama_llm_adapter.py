import httpx
from langchain_core.language_models import BaseChatModel
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional runtime fallback
    ChatOllama = None

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort


class OllamaLlmAdapter(LLMProviderPort):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        num_predict: int | None = 192,
        num_ctx: int | None = 4096,
        request_timeout_s: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.request_timeout_s = request_timeout_s
        chat_options: dict[str, object] = {
            "model": model,
            "base_url": self.base_url,
            "temperature": temperature,
            "reasoning": False,
            # Prevent proxy env vars from hijacking local/LAN Ollama calls.
            "async_client_kwargs": {"trust_env": False},
            "sync_client_kwargs": {"trust_env": False},
        }
        if num_predict is not None:
            chat_options["num_predict"] = num_predict
        if num_ctx is not None:
            chat_options["num_ctx"] = num_ctx
        self._chat = (
            ChatOllama(**chat_options)
            if ChatOllama
            else None
        )

    def get_chat_model(self) -> BaseChatModel:
        if self._chat is None:
            raise RuntimeError("Chat model is unavailable: install langchain-ollama and configure Ollama.")
        return self._chat

    async def generate_text(self, prompt: str) -> str:
        text = ""
        try:
            response = await self.get_chat_model().ainvoke(prompt)
            content = response.content
            if isinstance(content, str):
                text = content.strip()
            else:
                text = str(content).strip()
            if text:
                return text
        except Exception:
            # Fall back to raw Ollama REST call if langchain wrapper fails.
            pass

        try:
            options = {}
            if self.num_predict is not None:
                options["num_predict"] = self.num_predict
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    **options,
                },
            }
            async with httpx.AsyncClient(timeout=self.request_timeout_s, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
            text = str(body.get("response", "")).strip()
            if text:
                return text
        except Exception:
            pass
        return "Сейчас не удалось получить ответ от LLM, попробуй повторить запрос."

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code < 500
        except Exception:
            return False
