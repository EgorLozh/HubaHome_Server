import httpx
from langchain_core.language_models import BaseChatModel

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional runtime fallback
    ChatOpenAI = None

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort


class DeepSeekLlmAdapter(LLMProviderPort):
    name = "deepseek"

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._chat = None
        if ChatOpenAI and self.api_key:
            self._chat = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=0,
            )

    def get_chat_model(self) -> BaseChatModel:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        if self._chat is None:
            raise RuntimeError(
                "Chat model is unavailable: install langchain-openai and configure DeepSeek."
            )
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
            pass

        if not self.api_key:
            return "Сейчас не удалось получить ответ от LLM, попробуй повторить запрос."

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
            choices = body.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                text = str(message.get("content", "")).strip()
            if text:
                return text
        except Exception:
            pass
        return "Сейчас не удалось получить ответ от LLM, попробуй повторить запрос."

    async def ping(self) -> bool:
        if not self.api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=headers)
            return response.status_code < 500
        except Exception:
            return False
