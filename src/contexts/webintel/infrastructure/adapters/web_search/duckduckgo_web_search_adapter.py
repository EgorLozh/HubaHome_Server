import httpx

from src.contexts.webintel.application.ports.web_search_port import WebSearchPort


class DuckDuckGoWebSearchAdapter(WebSearchPort):
    name = "duckduckgo_web_search"

    async def search(self, query: str) -> list[dict]:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException:
            return [
                {
                    "source": "duckduckgo",
                    "snippet": "DuckDuckGo сейчас не отвечает по таймауту. Попробуйте повторить запрос позже.",
                }
            ]
        except httpx.HTTPError:
            return [
                {
                    "source": "duckduckgo",
                    "snippet": "DuckDuckGo временно недоступен. Попробуйте повторить запрос позже.",
                }
            ]
        except Exception:
            return [
                {
                    "source": "duckduckgo",
                    "snippet": "Веб-поиск временно недоступен. Попробуйте повторить запрос позже.",
                }
            ]

        results: list[dict] = []
        abstract_text = payload.get("AbstractText")
        if abstract_text:
            results.append(
                {
                    "source": payload.get("AbstractURL") or "duckduckgo",
                    "snippet": abstract_text,
                }
            )

        instant_results = payload.get("Results", [])[:3]
        for item in instant_results:
            if not isinstance(item, dict):
                continue
            text = item.get("Text")
            first_url = item.get("FirstURL")
            if isinstance(text, str) and text.strip():
                results.append({"source": first_url or "duckduckgo", "snippet": text})

        related_topics = payload.get("RelatedTopics", [])[:5]
        for topic in related_topics:
            if not isinstance(topic, dict):
                continue

            text = topic.get("Text")
            first_url = topic.get("FirstURL")
            if isinstance(text, str) and text.strip():
                results.append({"source": first_url or "duckduckgo", "snippet": text})
                continue

            nested_topics = topic.get("Topics")
            if not isinstance(nested_topics, list):
                continue
            for nested in nested_topics[:2]:
                if not isinstance(nested, dict):
                    continue
                nested_text = nested.get("Text")
                nested_url = nested.get("FirstURL")
                if isinstance(nested_text, str) and nested_text.strip():
                    results.append({"source": nested_url or "duckduckgo", "snippet": nested_text})

        if not results:
            results.append({"source": "duckduckgo", "snippet": "Релевантные данные по запросу не найдены."})
        return results
