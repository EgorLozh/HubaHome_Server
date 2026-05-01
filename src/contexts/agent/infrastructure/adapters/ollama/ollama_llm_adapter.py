import httpx
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional runtime fallback
    ChatOllama = None

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort


class OllamaLlmAdapter(LLMProviderPort):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._chat = ChatOllama(model=model, base_url=self.base_url, temperature=0) if ChatOllama else None

    async def generate_text(self, prompt: str) -> str:
        text = ""
        try:
            if self._chat is not None:
                response = await self._chat.ainvoke(prompt)
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
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
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
