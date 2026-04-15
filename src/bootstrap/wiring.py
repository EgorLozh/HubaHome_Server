from src.bootstrap.container import AppContainer
from src.contexts.agent.infrastructure.adapters.ollama.ollama_llm_adapter import (
    OllamaLlmAdapter,
)
from src.contexts.knowledge.infrastructure.adapters.qdrant.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)
from src.contexts.notification.infrastructure.adapters.stub_notification.stub_notification_adapter import (
    StubNotificationAdapter,
)
from src.contexts.speech.infrastructure.adapters.stub_stt.stub_speech_to_text_adapter import (
    StubSpeechToTextAdapter,
)
from src.contexts.speech.infrastructure.adapters.stub_tts.stub_text_to_speech_adapter import (
    StubTextToSpeechAdapter,
)
from src.contexts.webintel.infrastructure.adapters.stub_web_search.stub_web_search_adapter import (
    StubWebSearchAdapter,
)
from src.shared.config.settings import Settings


def build_container(settings: Settings) -> AppContainer:
    return AppContainer(
        settings=settings,
        llm_adapter=OllamaLlmAdapter(base_url=settings.ollama_url),
        vector_store_adapter=QdrantVectorStoreAdapter(base_url=settings.qdrant_url),
        speech_to_text_adapter=StubSpeechToTextAdapter(),
        text_to_speech_adapter=StubTextToSpeechAdapter(),
        web_search_adapter=StubWebSearchAdapter(),
        notification_adapter=StubNotificationAdapter(),
    )
