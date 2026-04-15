from typing import Literal, Union

from pydantic import BaseModel, Field


EventType = Literal[
    "wakeword_detected",
    "audio_chunk",
    "partial_transcript",
    "final_transcript",
    "assistant_text",
    "assistant_audio_chunk",
    "error",
]


class WsBaseEvent(BaseModel):
    event: EventType
    correlation_id: str | None = Field(default=None, alias="correlationId")

    model_config = {"populate_by_name": True}


class WakewordDetectedEvent(WsBaseEvent):
    event: Literal["wakeword_detected"]
    wake_word: str | None = Field(default="huba", alias="wakeWord")


class AudioChunkEvent(WsBaseEvent):
    event: Literal["audio_chunk"]
    chunk_id: int = Field(default=0, alias="chunkId")
    payload_b64: str | None = Field(default=None, alias="payloadB64")


class PartialTranscriptEvent(WsBaseEvent):
    event: Literal["partial_transcript"]
    text: str = ""


class FinalTranscriptEvent(WsBaseEvent):
    event: Literal["final_transcript"]
    text: str = ""


class AssistantTextEvent(WsBaseEvent):
    event: Literal["assistant_text"]
    text: str


class AssistantAudioChunkEvent(WsBaseEvent):
    event: Literal["assistant_audio_chunk"]
    chunk_id: int = Field(default=0, alias="chunkId")
    payload_b64: str = Field(default="", alias="payloadB64")


class ErrorEvent(WsBaseEvent):
    event: Literal["error"]
    message: str


IncomingWsEvent = Union[
    WakewordDetectedEvent,
    AudioChunkEvent,
    PartialTranscriptEvent,
    FinalTranscriptEvent,
    AssistantTextEvent,
    AssistantAudioChunkEvent,
    ErrorEvent,
]
