import asyncio
from types import SimpleNamespace

from src.contexts.conversation.application.use_cases.orchestrate_turn_agent import TurnOutput
from src.interfaces.websocket.message_dispatcher import VoiceSessionState, parse_event, stream_event_responses


class FakeSttAdapter:
    async def transcribe(self, audio_b64: str) -> str:
        return "распознанный текст"


class FakeTurnUseCase:
    async def execute(self, transcript: str) -> TurnOutput:
        return TurnOutput(assistant_text=f"Ответ: {transcript}", intent="chat")


class SlowTtsAdapter:
    def __init__(self) -> None:
        self.started = False

    async def synthesize(self, text: str) -> str:
        self.started = True
        await asyncio.sleep(0.01)
        return "encoded-audio"


def test_stream_event_responses_yields_text_before_tts_audio():
    async def run_test() -> None:
        tts_adapter = SlowTtsAdapter()
        container = SimpleNamespace(
            speech_to_text_adapter=FakeSttAdapter(),
            orchestrate_turn_use_case=FakeTurnUseCase(),
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

    asyncio.run(run_test())
