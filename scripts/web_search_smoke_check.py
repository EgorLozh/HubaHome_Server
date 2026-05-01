import argparse
import asyncio
import json
import sys
from urllib.parse import quote_plus

import httpx
from src.bootstrap.wiring import build_container
from src.shared.config.settings import get_settings


def _is_degraded_snippet(snippet: str) -> bool:
    lowered = snippet.lower()
    markers = (
        "временно недоступен",
        "таймаут",
        "не отвечает",
        "no web provider configured",
    )
    return any(marker in lowered for marker in markers)


async def run(query: str, limit: int) -> int:
    settings = get_settings()
    container = build_container(settings)

    normalized_query = query.strip()
    if not normalized_query:
        print("[fail] query is empty")
        return 1

    print(f"[info] searching web for: {normalized_query}")
    try:
        results = await container.web_search_adapter.search(normalized_query)
    except Exception as exc:
        print(f"[fail] web search request raised an exception: {exc}")
        return 1

    if not isinstance(results, list):
        print("[fail] adapter returned invalid response type")
        return 1
    if not results:
        print("[fail] adapter returned empty result list")
        return 1

    print(f"[ok] adapter={container.web_search_adapter.name}, results={len(results)}")
    shown = 0
    degraded = False
    for item in results:
        if shown >= max(limit, 1):
            break
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet", "")).strip()
        source = str(item.get("source", "unknown")).strip() or "unknown"
        if not snippet:
            continue
        shown += 1
        if _is_degraded_snippet(snippet):
            degraded = True
        print(f"[result {shown}] source={source}")
        print(f"  {snippet}")

    if shown == 0:
        print("[fail] no non-empty snippets in adapter response")
        return 1

    if degraded:
        print("[warn] provider returned degraded response")
        return 2

    print("[ok] web search smoke passed")
    return 0


async def run_raw_duckduckgo(query: str) -> int:
    normalized_query = query.strip()
    if not normalized_query:
        print("[fail] query is empty")
        return 1

    encoded_query = quote_plus(normalized_query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1"
    print(f"[info] raw request url: {url}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        print(f"[fail] raw request failed: {exc}")
        return 1

    print("[ok] raw response from DuckDuckGo:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct web search smoke check")
    parser.add_argument(
        "--query",
        default="weather izhevsk tomorrow",
        help="Search query for web provider",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many snippets to print",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw DuckDuckGo JSON response",
    )
    args = parser.parse_args()
    try:
        if args.raw:
            return asyncio.run(run_raw_duckduckgo(query=args.query))
        return asyncio.run(run(query=args.query, limit=args.limit))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
