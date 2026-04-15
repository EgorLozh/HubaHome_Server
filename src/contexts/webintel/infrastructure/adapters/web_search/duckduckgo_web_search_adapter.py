from urllib.parse import quote_plus

import httpx

from src.contexts.webintel.application.ports.web_search_port import WebSearchPort


class DuckDuckGoWebSearchAdapter(WebSearchPort):
    name = "duckduckgo_web_search"

    async def search(self, query: str) -> list[dict]:
        encoded_query = quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()

        results: list[dict] = []
        abstract_text = payload.get("AbstractText")
        if abstract_text:
            results.append(
                {
                    "source": payload.get("AbstractURL") or "duckduckgo",
                    "snippet": abstract_text,
                }
            )

        related_topics = payload.get("RelatedTopics", [])[:3]
        for topic in related_topics:
            text = topic.get("Text")
            first_url = topic.get("FirstURL")
            if text:
                results.append({"source": first_url or "duckduckgo", "snippet": text})

        if not results:
            results.append({"source": "duckduckgo", "snippet": "No relevant weather information found."})
        return results
