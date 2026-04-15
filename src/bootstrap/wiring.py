from src.bootstrap.container import AppContainer
from src.contexts.agent.infrastructure.adapters.ollama.ollama_llm_adapter import (
    OllamaLlmAdapter,
)
from src.contexts.conversation.application.use_cases.orchestrate_turn import (
    OrchestrateTurnUseCase,
)
from src.contexts.knowledge.infrastructure.adapters.qdrant.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)
from src.contexts.notification.infrastructure.adapters.stub_notification.stub_notification_adapter import (
    StubNotificationAdapter,
)
from src.contexts.speech.infrastructure.adapters.faster_whisper.faster_whisper_stt_adapter import (
    FasterWhisperSpeechToTextAdapter,
)
from src.contexts.speech.infrastructure.adapters.faster_whisper.whisper_model_manager import (
    WhisperModelManager,
)
from src.contexts.speech.infrastructure.adapters.piper.piper_text_to_speech_adapter import (
    PiperTextToSpeechAdapter,
)
from src.contexts.speech.infrastructure.adapters.piper.piper_model_manager import (
    PiperModelManager,
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
from src.contexts.webintel.infrastructure.adapters.web_search.duckduckgo_web_search_adapter import (
    DuckDuckGoWebSearchAdapter,
)
from src.shared.config.settings import Settings


def build_container(settings: Settings) -> AppContainer:
    llm_adapter = OllamaLlmAdapter(base_url=settings.ollama_url, model=settings.ollama_model)
    web_search_adapter = (
        DuckDuckGoWebSearchAdapter() if settings.enable_real_web_search else StubWebSearchAdapter()
    )
    speech_to_text_adapter = (
        FasterWhisperSpeechToTextAdapter(
            model_manager=WhisperModelManager(
                model_name=settings.whisper_model_name,
                models_dir=settings.whisper_models_dir,
                auto_download=settings.whisper_auto_download,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        )
        if settings.enable_real_stt
        else StubSpeechToTextAdapter()
    )
    piper_model_manager = PiperModelManager(
        model_name=settings.piper_model_name,
        models_dir=settings.piper_models_dir,
        auto_download=settings.piper_auto_download,
    )
    text_to_speech_adapter = (
        PiperTextToSpeechAdapter(
            model_manager=piper_model_manager,
            speaker_id=settings.piper_speaker_id,
            use_cuda=settings.piper_use_cuda,
        )
        if settings.enable_real_tts
        else StubTextToSpeechAdapter()
    )

    orchestrate_turn_use_case = OrchestrateTurnUseCase(
        llm_provider=llm_adapter,
        web_search=web_search_adapter,
    )

    return AppContainer(
        settings=settings,
        llm_adapter=llm_adapter,
        vector_store_adapter=QdrantVectorStoreAdapter(base_url=settings.qdrant_url),
        speech_to_text_adapter=speech_to_text_adapter,
        text_to_speech_adapter=text_to_speech_adapter,
        web_search_adapter=web_search_adapter,
        notification_adapter=StubNotificationAdapter(),
        orchestrate_turn_use_case=orchestrate_turn_use_case,
        whisper_model_status="pending" if settings.enable_real_stt else "not_configured",
        piper_model_status="pending" if settings.enable_real_tts else "not_configured",
    )
