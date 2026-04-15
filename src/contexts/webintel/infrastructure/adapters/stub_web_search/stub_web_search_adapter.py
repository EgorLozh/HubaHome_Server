from src.contexts.webintel.application.ports.web_search_port import WebSearchPort


class StubWebSearchAdapter(WebSearchPort):
    name = "stub_web_search"

    async def search(self, query: str) -> list[dict]:
        return [{"source": "stub", "snippet": f"No web provider configured yet for '{query}'"}]
