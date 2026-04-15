from dataclasses import dataclass

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort
from src.contexts.webintel.application.ports.web_search_port import WebSearchPort


@dataclass
class TurnOutput:
    assistant_text: str
    intent: str


class OrchestrateTurnUseCase:
    def __init__(self, llm_provider: LLMProviderPort, web_search: WebSearchPort) -> None:
        self.llm_provider = llm_provider
        self.web_search = web_search

    async def execute(self, transcript: str) -> TurnOutput:
        user_text = transcript.strip()
        if not user_text:
            return TurnOutput(assistant_text="Не расслышал команду. Повтори, пожалуйста.", intent="unknown")

        if self._is_weather_intent(user_text):
            search_results = await self.web_search.search(f"current weather {user_text}")
            snippets = "\n".join(f"- {item.get('snippet', '')}" for item in search_results[:3])
            prompt = (
                "Ты домашний ассистент. Сформируй краткий ответ по погоде на русском языке. "
                "Если данных недостаточно, попроси уточнить город.\n"
                f"Запрос пользователя: {user_text}\n"
                f"Данные поиска:\n{snippets}"
            )
            llm_answer = await self.llm_provider.generate_text(prompt)
            return TurnOutput(assistant_text=llm_answer.strip(), intent="weather")

        return TurnOutput(
            assistant_text="Пока во второй фазе я умею только сценарий погоды. Спроси про погоду.",
            intent="unsupported",
        )

    @staticmethod
    def _is_weather_intent(text: str) -> bool:
        lowered = text.lower()
        keywords = ("погод", "weather", "температур", "дожд", "снег", "ветер")
        return any(keyword in lowered for keyword in keywords)
