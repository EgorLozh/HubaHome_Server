import asyncio
from types import SimpleNamespace

from src.contexts.conversation.application.use_cases.orchestrate_turn_agent import TurnOutput
from src.interfaces.websocket.message_dispatcher import VoiceSessionState, parse_event, stream_event_responses


class FakeSttAdapter:
    async def transcribe(self, audio_b64: str) -> str:
        return "распознанный текст"


class FakeTurnUseCase:
    def __init__(self) -> None:
        self.last_transcript: str | None = None

    async def execute(self, transcript: str) -> TurnOutput:
        self.last_transcript = transcript
        return TurnOutput(assistant_text=f"Ответ: {transcript}", intent="chat")


class SlowTtsAdapter:
    def __init__(self) -> None:
        self.started = False

    async def synthesize(self, text: str) -> str:
        self.started = True
        await asyncio.sleep(0.01)
        return "encoded-audio"


class FailingSttAdapter:
    async def transcribe(self, audio_b64: str) -> str:
        raise RuntimeError("stt failed")


def test_stream_event_responses_yields_text_before_tts_audio():
    async def run_test() -> None:
        tts_adapter = SlowTtsAdapter()
        turn_use_case = FakeTurnUseCase()
        container = SimpleNamespace(
            speech_to_text_adapter=FakeSttAdapter(),
            orchestrate_turn_use_case=turn_use_case,
            text_to_speech_adapter=tts_adapter,
        )
        event = parse_event({"event": "final_transcript", "text": "привет"})
        session = VoiceSessionState(is_wakeword_detected=True)

        response_stream = stream_event_responses(event=event, session=session, container=container)
        first = await anext(response_stream)

        assert first == {"event": "assistant_text", "text": "Ответ: привет"}
        assert tts_adapter.started is False

        second = await anext(response_stream)
        assert second == {"event": "assistant_audio_chunk", "chunkId": 0, "payloadB64": "encoded-audio"}
        assert tts_adapter.started is True
        assert turn_use_case.last_transcript == "привет"

    asyncio.run(run_test())


def test_stream_event_responses_prefers_server_stt_over_client_text():
    async def run_test() -> None:
        turn_use_case = FakeTurnUseCase()
        container = SimpleNamespace(
            speech_to_text_adapter=FakeSttAdapter(),
            orchestrate_turn_use_case=turn_use_case,
            text_to_speech_adapter=SlowTtsAdapter(),
        )
        event = parse_event({"event": "final_transcript", "text": "client stub"})
        session = VoiceSessionState(is_wakeword_detected=True, audio_chunks_b64=["audio-b64"])

        response_stream = stream_event_responses(event=event, session=session, container=container)
        first = await anext(response_stream)

        assert first == {"event": "assistant_text", "text": "Ответ: распознанный текст"}
        assert turn_use_case.last_transcript == "распознанный текст"

    asyncio.run(run_test())


def test_stream_event_responses_falls_back_to_client_text_when_stt_fails():
    async def run_test() -> None:
        turn_use_case = FakeTurnUseCase()
        container = SimpleNamespace(
            speech_to_text_adapter=FailingSttAdapter(),
            orchestrate_turn_use_case=turn_use_case,
            text_to_speech_adapter=SlowTtsAdapter(),
        )
        event = parse_event({"event": "final_transcript", "text": "client fallback"})
        session = VoiceSessionState(is_wakeword_detected=True, audio_chunks_b64=["audio-b64"])

        response_stream = stream_event_responses(event=event, session=session, container=container)
        first = await anext(response_stream)

        assert first == {"event": "assistant_text", "text": "Ответ: client fallback"}
        assert turn_use_case.last_transcript == "client fallback"

    asyncio.run(run_test())
