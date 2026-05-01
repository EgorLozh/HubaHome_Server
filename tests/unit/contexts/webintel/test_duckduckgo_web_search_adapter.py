import asyncio

import httpx

from src.contexts.webintel.infrastructure.adapters.web_search.duckduckgo_web_search_adapter import (
    DuckDuckGoWebSearchAdapter,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _SuccessClient:
    def __init__(self, payload: dict, timeout: float) -> None:
        self._payload = payload
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(self._payload)


class _TimeoutClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        raise httpx.TimeoutException("timeout")


class _HttpErrorClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        raise httpx.HTTPError("http error")


def test_duckduckgo_search_parses_abstract_and_nested_topics(monkeypatch):
    payload = {
        "AbstractText": "Current weather in Izhevsk.",
        "AbstractURL": "https://example.com/weather",
        "RelatedTopics": [
            {
                "Name": "Weather topics",
                "Topics": [
                    {"Text": "Forecast for next days", "FirstURL": "https://example.com/forecast"},
                    {"Text": "Humidity levels", "FirstURL": "https://example.com/humidity"},
                ],
            }
        ],
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _SuccessClient(payload=payload, timeout=timeout))

    adapter = DuckDuckGoWebSearchAdapter()
    result = asyncio.run(adapter.search("weather izhevsk"))

    assert len(result) >= 2
    assert "Izhevsk" in result[0]["snippet"]
    assert any("Forecast" in item["snippet"] for item in result)


def test_duckduckgo_search_handles_timeout(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _TimeoutClient(timeout=timeout))

    adapter = DuckDuckGoWebSearchAdapter()
    result = asyncio.run(adapter.search("weather izhevsk"))

    assert len(result) == 1
    assert "таймауту" in result[0]["snippet"]


def test_duckduckgo_search_handles_http_error(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: _HttpErrorClient(timeout=timeout))

    adapter = DuckDuckGoWebSearchAdapter()
    result = asyncio.run(adapter.search("weather izhevsk"))

    assert len(result) == 1
    assert "временно недоступен" in result[0]["snippet"]
