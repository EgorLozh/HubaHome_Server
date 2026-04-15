from pydantic import ValidationError

from src.shared.contracts.ws_messages import (
    AssistantAudioChunkEvent,
    AssistantTextEvent,
    ErrorEvent,
    IncomingWsEvent,
)
from src.shared.observability.metrics import ERROR_COUNT, WS_EVENT_COUNT


def parse_event(payload: dict) -> IncomingWsEvent:
    event_type = payload.get("event", "error")
    WS_EVENT_COUNT.labels(event_type=event_type).inc()

    try:
        if event_type == "wakeword_detected":
            from src.shared.contracts.ws_messages import WakewordDetectedEvent

            return WakewordDetectedEvent.model_validate(payload)
        if event_type == "audio_chunk":
            from src.shared.contracts.ws_messages import AudioChunkEvent

            return AudioChunkEvent.model_validate(payload)
        if event_type == "partial_transcript":
            from src.shared.contracts.ws_messages import PartialTranscriptEvent

            return PartialTranscriptEvent.model_validate(payload)
        if event_type == "final_transcript":
            from src.shared.contracts.ws_messages import FinalTranscriptEvent

            return FinalTranscriptEvent.model_validate(payload)
        if event_type == "assistant_text":
            return AssistantTextEvent.model_validate(payload)
        if event_type == "assistant_audio_chunk":
            return AssistantAudioChunkEvent.model_validate(payload)
        return ErrorEvent(event="error", message="Unknown event type")
    except ValidationError as error:
        ERROR_COUNT.labels(source="ws_validation").inc()
        return ErrorEvent(event="error", message=f"Invalid payload: {error.errors()}")


def handle_event(event: IncomingWsEvent) -> list[dict]:
    if event.event == "wakeword_detected":
        return [{"event": "assistant_text", "text": "Слушаю, говори команду."}]

    if event.event == "audio_chunk":
        return [{"event": "assistant_audio_chunk", "chunkId": event.chunk_id, "payloadB64": ""}]

    if event.event == "partial_transcript":
        return [{"event": "assistant_text", "text": f"Понял частично: {event.text}"}]

    if event.event == "final_transcript":
        return [{"event": "assistant_text", "text": f"Phase 1 echo: {event.text}"}]

    if event.event == "assistant_text":
        return [event.model_dump(by_alias=True)]

    if event.event == "assistant_audio_chunk":
        return [event.model_dump(by_alias=True)]

    if event.event == "error":
        ERROR_COUNT.labels(source="ws_error_event").inc()
        return [event.model_dump(by_alias=True)]

    ERROR_COUNT.labels(source="ws_unknown").inc()
    return [{"event": "error", "message": "Unhandled event"}]
