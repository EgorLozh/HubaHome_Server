import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.bootstrap.container import AppContainer
from src.bootstrap.wiring import build_container
from src.contexts.speech.infrastructure.adapters.stub_stt.stub_speech_to_text_adapter import (
    StubSpeechToTextAdapter,
)
from src.contexts.speech.infrastructure.adapters.stub_tts.stub_text_to_speech_adapter import (
    StubTextToSpeechAdapter,
)
from src.interfaces.http.api_router import build_api_router
from src.interfaces.websocket.voice_session_router import build_ws_router
from src.shared.config.settings import Settings, get_settings
from src.shared.logging.logger import configure_logging, get_logger
from src.shared.observability.metrics import ERROR_COUNT, HttpMetricsMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if container.settings.enable_real_stt:
            logger = get_logger(__name__)
            if not hasattr(container.speech_to_text_adapter, "initialize"):
                container.whisper_model_status = "unavailable_adapter"
            else:
                try:
                    status = await asyncio.to_thread(container.speech_to_text_adapter.initialize)
                    container.whisper_model_status = status
                    logger.info("Whisper model is ready with status: %s", status)
                except Exception as error:
                    ERROR_COUNT.labels(source="stt_startup").inc()
                    container.whisper_model_status = "error"
                    container.speech_to_text_adapter = StubSpeechToTextAdapter()
                    logger.exception(
                        "Whisper startup initialization failed; fallback to stub: %s", error
                    )

        if container.settings.enable_real_tts:
            logger = get_logger(__name__)
            if not hasattr(container.text_to_speech_adapter, "initialize"):
                container.piper_model_status = "unavailable_adapter"
            else:
                try:
                    status = await asyncio.to_thread(container.text_to_speech_adapter.initialize)
                    container.piper_model_status = status
                    logger.info("Piper model is ready with status: %s", status)
                except Exception as error:
                    ERROR_COUNT.labels(source="tts_startup").inc()
                    container.piper_model_status = "error"
                    container.text_to_speech_adapter = StubTextToSpeechAdapter()
                    logger.exception(
                        "Piper startup initialization failed; fallback to stub: %s", error
                    )
        yield

    app = FastAPI(title=active_settings.app_name, lifespan=lifespan)
    app.add_middleware(HttpMetricsMiddleware)

    container = build_container(active_settings)
    app.state.container = container

    app.include_router(build_api_router(container))
    app.include_router(build_ws_router(container))
    return app
