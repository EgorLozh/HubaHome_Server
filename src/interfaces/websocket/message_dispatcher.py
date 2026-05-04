import asyncio
import base64
import io
import wave
from dataclasses import dataclass, field
from time import perf_counter

from pydantic import ValidationError

from src.bootstrap.container import AppContainer
from src.interfaces.voice.turn_runtime import run_voice_turn
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
    audio_chunks: list["BufferedAudioChunk"] = field(default_factory=list)


@dataclass
class BufferedAudioChunk:
    chunk_id: int
    payload_b64: str


def _merge_audio_chunks(audio_chunks: list[BufferedAudioChunk]) -> str:
    ordered_chunks = sorted(
        (chunk for chunk in audio_chunks if chunk.payload_b64.strip()),
        key=lambda chunk: chunk.chunk_id,
    )
    if not ordered_chunks:
        return ""
    if len(ordered_chunks) == 1:
        return ordered_chunks[0].payload_b64.strip()

    pcm_parts: list[bytes] = []
    sample_rate = 16_000
    channels = 1
    sample_width = 2

    for chunk in ordered_chunks:
        audio_bytes = base64.b64decode(chunk.payload_b64)
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if not pcm_parts:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            pcm_parts.append(wav_file.readframes(wav_file.getnframes()))

    if not pcm_parts:
        return ""

    merged_buffer = io.BytesIO()
    with wave.open(merged_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(pcm_parts))
    return base64.b64encode(merged_buffer.getvalue()).decode("ascii")


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


async def handle_event(
    event: IncomingWsEvent,
    session: VoiceSessionState,
    container: AppContainer,
) -> list[dict]:
    if event.event == "wakeword_detected":
        session.is_wakeword_detected = True
        session.audio_chunks.clear()
        return []

    if event.event == "audio_chunk":
        if len(event.payload_b64 or "") > 2_000_000:
            ERROR_COUNT.labels(source="ws_payload_too_large").inc()
            return [{"event": "error", "message": "Audio chunk is too large"}]
        session.audio_chunks.append(
            BufferedAudioChunk(
                chunk_id=event.chunk_id,
                payload_b64=event.payload_b64 or "",
            )
        )
        return []

    if event.event == "partial_transcript":
        return [{"event": "assistant_text", "text": f"Понял частично: {event.text}"}]

    if event.event == "final_transcript":
        if not session.is_wakeword_detected:
            ERROR_COUNT.labels(source="ws_wakeword_missing").inc()
            return [{"event": "error", "message": "Wakeword was not detected"}]

        started_stt = perf_counter()
        client_transcript = event.text.strip()
        transcript = ""
        merged_audio_chunk = _merge_audio_chunks(session.audio_chunks)
        if merged_audio_chunk and container.speech_to_text_adapter.name != "stub_stt":
            try:
                transcript = await asyncio.wait_for(
                    container.speech_to_text_adapter.transcribe(merged_audio_chunk),
                    timeout=container.settings.stt_timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                ERROR_COUNT.labels(source="stt_timeout").inc()
            except Exception:
                ERROR_COUNT.labels(source="stt").inc()
            finally:
                observe_stage_latency(stage="stt", started=started_stt)
        else:
            observe_stage_latency(stage="stt", started=started_stt)

        normalized_transcript = transcript.strip()
        if normalized_transcript.lower() == "stub transcription":
            ERROR_COUNT.labels(source="stt_stub_result").inc()
            normalized_transcript = ""

        if not normalized_transcript and client_transcript:
            if merged_audio_chunk:
                ERROR_COUNT.labels(source="stt_text_fallback").inc()
            normalized_transcript = client_transcript

        transcript = normalized_transcript

        turn_result = await run_voice_turn(container=container, transcript=transcript)
        responses: list[dict] = [{"event": "assistant_text", "text": turn_result.assistant_text}]
        if turn_result.assistant_audio_b64:
            responses.append(
                {"event": "assistant_audio_chunk", "chunkId": 0, "payloadB64": turn_result.assistant_audio_b64}
            )

        session.audio_chunks.clear()
        return responses

    if event.event == "assistant_text":
        return [event.model_dump(by_alias=True)]

    if event.event == "assistant_audio_chunk":
        return [event.model_dump(by_alias=True)]

    if event.event == "error":
        ERROR_COUNT.labels(source="ws_error_event").inc()
        return [event.model_dump(by_alias=True)]

    ERROR_COUNT.labels(source="ws_unknown").inc()
    return [{"event": "error", "message": "Unhandled event"}]
