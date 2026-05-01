import argparse
import asyncio
import sys
from typing import Any

from src.bootstrap.wiring import build_container
from src.shared.config.settings import get_settings


async def run(full: bool) -> int:
    settings = get_settings()
    container = build_container(settings)
    use_case = container.orchestrate_turn_use_case

    print("[info] checking metadata_instruction_tool...")
    set_result = await use_case.tool_registry.call(
        "metadata_instruction_tool",
        {"action": "set_instruction", "key": "name", "value": "Егор"},
    )
    if not set_result.ok:
        print("[fail] metadata set failed")
        print(f"[debug] {set_result.message}")
        return 1
    print(f"[ok] metadata set: {set_result.message}")

    get_result = await use_case.tool_registry.call(
        "metadata_instruction_tool",
        {"action": "get_instruction", "key": "name"},
    )
    if not get_result.ok:
        print("[fail] metadata get failed")
        print(f"[debug] {get_result.message}")
        return 1
    print(f"[ok] metadata get: {get_result.message}")

    if not full:
        print("[ok] basic tools smoke checks passed")
        return 0

    print("[info] full mode enabled: checking knowledge_document_tool...")
    vector_ok = await container.vector_store_adapter.ping()
    if not vector_ok:
        print("[fail] Qdrant is unreachable. Full mode requires vector store.")
        return 1

    create_result = await use_case.tool_registry.call(
        "knowledge_document_tool",
        {
            "action": "create_document",
            "title": "Тестовый документ",
            "text": "молоко и хлеб",
            "tags": ["smoke"],
        },
    )
    if not create_result.ok:
        print("[fail] document create failed")
        print(f"[debug] {create_result.message}")
        return 1
    document = create_result.data.get("document", {})
    doc_id = str(document.get("id", ""))
    if not doc_id:
        print("[fail] document create returned empty id")
        return 1
    print(f"[ok] document create: id={doc_id}")

    search_result = await use_case.tool_registry.call(
        "knowledge_document_tool",
        {"action": "search_documents", "query": "молоко"},
    )
    if not search_result.ok:
        print("[fail] document search failed")
        print(f"[debug] {search_result.message}")
        return 1
    items = _safe_items(search_result.data)
    if not items:
        print("[fail] document search returned no items")
        return 1
    print(f"[ok] document search: {len(items)} item(s)")

    update_result = await use_case.tool_registry.call(
        "knowledge_document_tool",
        {"action": "update_document", "id": doc_id, "text": "молоко, хлеб и сыр"},
    )
    if not update_result.ok:
        print("[fail] document update failed")
        print(f"[debug] {update_result.message}")
        return 1
    print("[ok] document update passed")

    delete_result = await use_case.tool_registry.call(
        "knowledge_document_tool",
        {"action": "delete_document", "id": doc_id},
    )
    if not delete_result.ok:
        print("[fail] document delete failed")
        print(f"[debug] {delete_result.message}")
        return 1
    print("[ok] document delete passed")

    print("[info] checking weather_tool...")
    weather_result = await use_case.tool_registry.call(
        "weather_tool",
        {"query": "москва"},
    )
    if weather_result.ok:
        print("[ok] weather tool passed")
    else:
        # Weather may fail due to external dependency/network restrictions.
        print(f"[warn] weather tool failed: {weather_result.message}")

    print("[ok] full tools smoke checks passed")
    return 0


def _safe_items(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("items", [])
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for tool registry and tool execution")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run metadata + knowledge CRUD + weather checks (requires Qdrant)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(full=args.full))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
