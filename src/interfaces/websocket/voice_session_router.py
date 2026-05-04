from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.bootstrap.container import AppContainer
from src.interfaces.websocket.handshake import is_authorized, resolve_api_key
from src.interfaces.websocket.message_dispatcher import VoiceSessionState, handle_event, parse_event
from src.shared.logging.correlation import set_correlation_id
from src.shared.observability.metrics import ERROR_COUNT


def build_ws_router(container: AppContainer) -> APIRouter:
    router = APIRouter()

    @router.websocket("/v1/voice/session")
    async def voice_session(websocket: WebSocket) -> None:
        provided_key = resolve_api_key(websocket)
        if not is_authorized(provided_key, container.settings.api_key):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        set_correlation_id(websocket.headers.get("x-correlation-id"))
        session = VoiceSessionState()

        try:
            while True:
                payload = await websocket.receive_json()
                event = parse_event(payload)
                responses = await handle_event(event=event, session=session, container=container)
                for response in responses:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            return
        except Exception:
            ERROR_COUNT.labels(source="ws_runtime").inc()
            try:
                await websocket.send_json({"event": "error", "message": "Internal server error"})
            except Exception:
                pass
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

    return router
