from fastapi import APIRouter, Header, HTTPException, status

from src.bootstrap.container import AppContainer
from src.interfaces.voice.turn_runtime import run_voice_turn
from src.interfaces.websocket.handshake import is_authorized
from src.shared.contracts.http_voice import VoiceTextRequest, VoiceTextResponse
from src.shared.logging.correlation import set_correlation_id


def build_voice_text_router(container: AppContainer) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/voice/text", response_model=VoiceTextResponse)
    async def voice_text(
        payload: VoiceTextRequest,
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> VoiceTextResponse:
        if not is_authorized(x_api_key or "", container.settings.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        set_correlation_id(payload.correlation_id)
        result = await run_voice_turn(container=container, transcript=payload.text.strip())
        return VoiceTextResponse(
            assistantText=result.assistant_text,
            assistantAudioB64=result.assistant_audio_b64,
            intent=result.intent,
            correlationId=payload.correlation_id,
        )

    return router
