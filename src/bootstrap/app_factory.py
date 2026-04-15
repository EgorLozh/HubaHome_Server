from fastapi import FastAPI

from src.bootstrap.container import AppContainer
from src.bootstrap.wiring import build_container
from src.interfaces.http.api_router import build_api_router
from src.interfaces.websocket.voice_session_router import build_ws_router
from src.shared.config.settings import Settings, get_settings
from src.shared.logging.logger import configure_logging
from src.shared.observability.metrics import HttpMetricsMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level)

    app = FastAPI(title=active_settings.app_name)
    app.add_middleware(HttpMetricsMiddleware)

    container = build_container(active_settings)
    app.state.container = container

    app.include_router(build_api_router(container))
    app.include_router(build_ws_router(container))
    return app
