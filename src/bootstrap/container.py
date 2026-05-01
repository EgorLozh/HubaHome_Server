from dataclasses import dataclass

from src.contexts.agent.application.ports.llm_provider_port import LLMProviderPort
from src.contexts.conversation.application.use_cases.orchestrate_turn import (
    OrchestrateTurnUseCase,
)
from src.contexts.knowledge.application.ports.vector_store_port import VectorStorePort
from src.contexts.notification.application.ports.notification_port import NotificationPort
from src.contexts.speech.application.ports.speech_to_text_port import SpeechToTextPort
from src.contexts.speech.application.ports.text_to_speech_port import TextToSpeechPort
from src.contexts.webintel.application.ports.web_search_port import WebSearchPort
from src.shared.config.settings import Settings


@dataclass
class AppContainer:
    settings: Settings
    llm_adapter: LLMProviderPort
    vector_store_adapter: VectorStorePort
    speech_to_text_adapter: SpeechToTextPort
    text_to_speech_adapter: TextToSpeechPort
    web_search_adapter: WebSearchPort
    notification_adapter: NotificationPort
    orchestrate_turn_use_case: OrchestrateTurnUseCase
    whisper_model_status: str = "not_configured"
    piper_model_status: str = "not_configured"
