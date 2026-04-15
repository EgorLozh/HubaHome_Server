import httpx

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort


class OllamaLlmAdapter(LLMProviderPort):
    name = "ollama"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def generate_text(self, prompt: str) -> str:
        return f"Ollama stub response for: {prompt}"

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code < 500
        except Exception:
            return False
