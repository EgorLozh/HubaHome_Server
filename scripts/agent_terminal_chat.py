import argparse
import asyncio
import time

from src.bootstrap.wiring import build_container
from src.shared.config.settings import get_settings


DEMO_SCENARIOS: dict[str, list[str]] = {
    "basic": [
        "Скажи коротко, что ты онлайн.",
        "меня зовут Егор",
        "как меня зовут",
    ],
    "tools": [
        "какая погода в Москве",
        "создай документ с заголовком Покупки и текстом молоко и хлеб",
        "найди документ про молоко",
        "обнови документ: добавь сыр",
        "удали документ про молоко",
    ],
}


def _print_help() -> None:
    print(
        "\nКоманды:\n"
        "  /help                - показать помощь\n"
        "  /quit                - выйти\n"
        "  /scenario basic      - прогнать базовый сценарий через агента\n"
        "  /scenario tools      - прогнать сценарий с разными tools через агента\n"
        "  /config              - показать ключевые runtime настройки\n"
    )
    print(
        "\nПример фраз для ручного теста:\n"
        "  - какая погода в москве\n"
        "  - меня зовут Егор\n"
        "  - как меня зовут\n"
        "  - создай документ с заголовком Заметка и текстом молоко и хлеб\n"
        "  - найди документ про молоко\n"
        "  - удали документ про молоко\n"
    )


async def _run_scenario(use_case, scenario_name: str) -> None:
    phrases = DEMO_SCENARIOS.get(scenario_name)
    if not phrases:
        print(f"[warn] unknown scenario: {scenario_name}")
        print(f"[hint] available: {', '.join(sorted(DEMO_SCENARIOS.keys()))}")
        return

    print(f"[info] running scenario: {scenario_name}")
    for phrase in phrases:
        print(f"\nyou> {phrase}")
        started = time.perf_counter()
        turn = await use_case.execute(phrase)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"bot> [{turn.intent}] {turn.assistant_text}")
        print(f"[time] {elapsed_ms} ms")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive terminal chat with agent runtime")
    parser.add_argument(
        "--scenario",
        choices=sorted(DEMO_SCENARIOS.keys()),
        help="Run a predefined scenario and exit",
    )
    args = parser.parse_args()

    settings = get_settings()
    container = build_container(settings)
    use_case = container.orchestrate_turn_use_case

    llm_ok = await container.llm_adapter.ping()
    if not llm_ok:
        print("[fail] Ollama is unreachable. Check OLLAMA_URL and running service.")
        return 1

    if args.scenario:
        await _run_scenario(use_case=use_case, scenario_name=args.scenario)
        return 0

    print("[ok] agent terminal chat started")
    print("[info] type /help for commands")
    print("[info] all messages are passed as transcript to OrchestrateTurnUseCase")

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] exit")
            return 0

        if not user_text:
            continue
        if user_text == "/quit":
            print("[info] exit")
            return 0
        if user_text == "/help":
            _print_help()
            continue
        if user_text == "/config":
            print(
                f"[config] model={settings.ollama_model}, url={settings.ollama_url}, "
                f"maxSteps={settings.agent_max_steps_per_turn}, "
                f"maxTools={settings.agent_max_tool_calls_per_turn}, "
                f"timeoutMs={settings.agent_turn_timeout_ms}"
            )
            continue
        if user_text.startswith("/scenario"):
            parts = user_text.split(maxsplit=1)
            scenario_name = parts[1] if len(parts) == 2 else ""
            await _run_scenario(use_case=use_case, scenario_name=scenario_name)
            continue

        started = time.perf_counter()
        turn = await use_case.execute(user_text)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"bot> [{turn.intent}] {turn.assistant_text}")
        print(f"[time] {elapsed_ms} ms")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
