import argparse
import asyncio
import sys

from src.bootstrap.wiring import build_container
from src.shared.config.settings import get_settings


async def run(full: bool) -> int:
    settings = get_settings()
    container = build_container(settings)
    use_case = container.orchestrate_turn_use_case

    print("[info] checking metadata instruction flow via agent...")
    set_result = await use_case.execute("меня зовут Егор")
    if set_result.intent not in {"metadata", "chat"}:
        print("[fail] metadata set flow failed")
        print(f"[debug] intent={set_result.intent!r}, text={set_result.assistant_text!r}")
        return 1
    print(f"[ok] metadata set: intent={set_result.intent}, text={set_result.assistant_text}")

    get_result = await use_case.execute("как меня зовут")
    if not get_result.assistant_text.strip():
        print("[fail] metadata get flow returned empty response")
        return 1
    print(f"[ok] metadata get: intent={get_result.intent}, text={get_result.assistant_text}")

    if not full:
        print("[ok] basic tools smoke checks passed")
        return 0

    print("[info] full mode enabled: checking knowledge/document and weather flows...")
    vector_ok = await container.vector_store_adapter.ping()
    if not vector_ok:
        print("[fail] Qdrant is unreachable. Full mode requires vector store.")
        return 1

    create_turn = await use_case.execute("создай документ с заголовком Тест и текстом молоко и хлеб")
    if create_turn.intent not in {"knowledge", "chat"}:
        print("[fail] document create flow failed")
        print(f"[debug] intent={create_turn.intent!r}, text={create_turn.assistant_text!r}")
        return 1
    print(f"[ok] document create: intent={create_turn.intent}")

    # A direct metadata lookup is used for deterministic CRUD checks in smoke mode.
    all_docs = use_case.metadata_store.search(query="молоко", limit=1)
    if not all_docs:
        print("[fail] document create produced no metadata records")
        return 1
    doc_id = str(all_docs[0].get("id", "")).strip()
    if not doc_id:
        print("[fail] created document has no id")
        return 1

    update_result = await use_case._update_document_tool(id=doc_id, text="молоко, хлеб и сыр")
    if "обновлен" not in update_result.lower():
        print("[fail] document update failed")
        print(f"[debug] result={update_result!r}")
        return 1
    print("[ok] document update passed")

    search_turn = await use_case.execute("найди документ про молоко")
    if not search_turn.assistant_text.strip():
        print("[fail] document search returned empty text")
        return 1
    print(f"[ok] document search: intent={search_turn.intent}, text={search_turn.assistant_text}")

    delete_result = await use_case._delete_document_tool(id=doc_id)
    if "удален" not in delete_result.lower():
        print("[fail] document delete failed")
        print(f"[debug] result={delete_result!r}")
        return 1
    print("[ok] document delete passed")

    weather_turn = await use_case.execute("какая погода в москве")
    if weather_turn.assistant_text.strip():
        print(f"[ok] weather flow: intent={weather_turn.intent}, text={weather_turn.assistant_text}")
    else:
        print("[warn] weather flow returned empty text")

    print("[ok] full tools smoke checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for LangChain agent tool execution")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run metadata + knowledge + weather checks (requires Qdrant)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(full=args.full))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
