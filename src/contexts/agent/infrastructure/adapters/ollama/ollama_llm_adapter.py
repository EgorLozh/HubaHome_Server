import httpx

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort


class OllamaLlmAdapter(LLMProviderPort):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_text(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
            text = body.get("response", "").strip()
            if text:
                return text
        except Exception:
            pass
        return "Сейчас не удалось получить ответ от LLM, попробуй повторить запрос."

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code < 500
        except Exception:
            return False
