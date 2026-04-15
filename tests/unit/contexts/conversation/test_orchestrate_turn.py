import asyncio

from src.contexts.conversation.application.use_cases.orchestrate_turn import OrchestrateTurnUseCase


class FakeLlm:
    async def generate_text(self, prompt: str) -> str:
        return "Сегодня около +12, облачно."

    async def ping(self) -> bool:
        return True


class FakeWebSearch:
    async def search(self, query: str) -> list[dict]:
        return [{"source": "test", "snippet": f"Weather data for query: {query}"}]


def test_orchestrate_turn_weather_intent():
    use_case = OrchestrateTurnUseCase(llm_provider=FakeLlm(), web_search=FakeWebSearch())
    result = asyncio.run(use_case.execute("какая погода в москве"))

    assert result.intent == "weather"
    assert "облачно" in result.assistant_text


def test_orchestrate_turn_unsupported_intent():
    use_case = OrchestrateTurnUseCase(llm_provider=FakeLlm(), web_search=FakeWebSearch())
    result = asyncio.run(use_case.execute("поставь музыку"))

    assert result.intent == "unsupported"
    assert "погоды" in result.assistant_text
