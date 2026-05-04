from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter

from pydantic import ValidationError

from src.bootstrap.container import AppContainer
from src.shared.contracts.ws_messages import (
    AssistantAudioChunkEvent,
    AssistantTextEvent,
    ErrorEvent,
    IncomingWsEvent,
)
from src.shared.observability.metrics import ERROR_COUNT, WS_EVENT_COUNT, observe_stage_latency


@dataclass
class VoiceSessionState:
    is_wakeword_detected: bool = False
    audio_chunks_b64: list[str] = field(default_factory=list)


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


async def stream_event_responses(
    event: IncomingWsEvent,
    session: VoiceSessionState,
    container: AppContainer,
) -> AsyncIterator[dict]:
    if event.event == "wakeword_detected":
        session.is_wakeword_detected = True
        session.audio_chunks_b64.clear()
        yield {"event": "assistant_text", "text": "Слушаю, говори команду."}
        return

    if event.event == "audio_chunk":
        if len(event.payload_b64 or "") > 2_000_000:
            ERROR_COUNT.labels(source="ws_payload_too_large").inc()
            yield {"event": "error", "message": "Audio chunk is too large"}
            return
        session.audio_chunks_b64.append(event.payload_b64 or "")
        return

    if event.event == "partial_transcript":
        yield {"event": "assistant_text", "text": f"Понял частично: {event.text}"}
        return

    if event.event == "final_transcript":
        if not session.is_wakeword_detected:
            ERROR_COUNT.labels(source="ws_wakeword_missing").inc()
            yield {"event": "error", "message": "Wakeword was not detected"}
            return

        started_stt = perf_counter()
        transcript = event.text.strip()
        if not transcript and session.audio_chunks_b64:
            try:
                transcript = await container.speech_to_text_adapter.transcribe(session.audio_chunks_b64[-1])
            except Exception:
                ERROR_COUNT.labels(source="stt").inc()
                transcript = ""
            finally:
                observe_stage_latency(stage="stt", started=started_stt)
        else:
            observe_stage_latency(stage="stt", started=started_stt)

        started_agent = perf_counter()
        try:
            turn = await container.orchestrate_turn_use_case.execute(transcript)
            assistant_text = turn.assistant_text
        except Exception:
            ERROR_COUNT.labels(source="agent").inc()
            assistant_text = "Не удалось обработать запрос. Попробуй еще раз."
        finally:
            observe_stage_latency(stage="agent", started=started_agent)

        yield {"event": "assistant_text", "text": assistant_text}

        started_tts = perf_counter()
        try:
            audio_b64 = await container.text_to_speech_adapter.synthesize(assistant_text)
            if audio_b64:
                yield {"event": "assistant_audio_chunk", "chunkId": 0, "payloadB64": audio_b64}
        except Exception:
            ERROR_COUNT.labels(source="tts").inc()
        finally:
            observe_stage_latency(stage="tts", started=started_tts)

        session.audio_chunks_b64.clear()
        return

    if event.event == "assistant_text":
        yield event.model_dump(by_alias=True)
        return

    if event.event == "assistant_audio_chunk":
        yield event.model_dump(by_alias=True)
        return

    if event.event == "error":
        ERROR_COUNT.labels(source="ws_error_event").inc()
        yield event.model_dump(by_alias=True)
        return

    ERROR_COUNT.labels(source="ws_unknown").inc()
    yield {"event": "error", "message": "Unhandled event"}


async def handle_event(
    event: IncomingWsEvent,
    session: VoiceSessionState,
    container: AppContainer,
) -> list[dict]:
    return [response async for response in stream_event_responses(event=event, session=session, container=container)]
