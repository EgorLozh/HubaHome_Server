import asyncio
import argparse
import sys

from src.bootstrap.wiring import build_container
from src.shared.config.settings import get_settings


FAILURE_HINT = "Сейчас не удалось получить ответ от LLM, попробуй повторить запрос."


async def run(full: bool) -> int:
    settings = get_settings()
    container = build_container(settings)

    print("[info] checking LLM adapter connectivity...")
    llm_ok = await container.llm_adapter.ping()
    if not llm_ok:
        print("[fail] Ollama is unreachable. Check OLLAMA_URL and running service.")
        return 1
    print("[ok] Ollama ping passed")

    print("[info] running direct LLM generation check...")
    direct_answer = await container.llm_adapter.generate_text(
        "Ответь по-русски одним словом: готово."
    )
    if not direct_answer.strip() or FAILURE_HINT in direct_answer:
        print("[fail] direct LLM generation failed")
        print(f"[debug] response: {direct_answer!r}")
        return 1
    print(f"[ok] direct answer: {direct_answer.strip()}")

    print("[info] checking web search provider...")
    web_results = await container.web_search_adapter.search("current weather izhevsk")
    if not isinstance(web_results, list):
        print("[fail] web search adapter returned invalid result type")
        return 1
    if not web_results:
        print("[fail] web search adapter returned empty result list")
        return 1
    first_snippet = str(web_results[0].get("snippet", "")).strip() if isinstance(web_results[0], dict) else ""
    if "недоступен" in first_snippet.lower() or "таймаут" in first_snippet.lower():
        print(f"[warn] web search provider degraded: {first_snippet}")
    else:
        print("[ok] web search provider returned data")

    if not full:
        print("[ok] basic LLM smoke checks passed")
        return 0

    print("[info] full mode enabled: running orchestrator checks...")
    print("[info] running orchestrator chat check...")
    turn = await container.orchestrate_turn_use_case.execute("Скажи коротко, что ты онлайн.")
    if not turn.assistant_text.strip() or turn.intent in {"error", "timeout"}:
        print("[fail] orchestrator chat check failed")
        print(f"[debug] turn: intent={turn.intent!r}, text={turn.assistant_text!r}")
        print("[hint] Increase AGENT_TURN_TIMEOUT_MS or use a faster local model for full checks.")
        return 1
    print(f"[ok] orchestrator chat: intent={turn.intent}, text={turn.assistant_text}")

    print("[info] running metadata instruction check...")
    set_name_turn = await container.orchestrate_turn_use_case.execute("меня зовут Егор")
    if set_name_turn.intent not in {"metadata", "chat"}:
        print("[fail] metadata set flow returned unexpected intent")
        print(f"[debug] turn: intent={set_name_turn.intent!r}, text={set_name_turn.assistant_text!r}")
        return 1
    print(f"[ok] metadata set: intent={set_name_turn.intent}, text={set_name_turn.assistant_text}")

    read_name_turn = await container.orchestrate_turn_use_case.execute("как меня зовут")
    if not read_name_turn.assistant_text.strip():
        print("[fail] metadata read flow returned empty text")
        return 1
    print(f"[ok] metadata read: intent={read_name_turn.intent}, text={read_name_turn.assistant_text}")

    print("[ok] LLM smoke checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for LLM and agent runtime")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run extended orchestrator + metadata checks in addition to basic LLM checks",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(full=args.full))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
