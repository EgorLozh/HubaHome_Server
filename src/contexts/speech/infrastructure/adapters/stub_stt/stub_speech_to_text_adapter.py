from src.contexts.speech.application.ports.speech_to_text_port import SpeechToTextPort


class StubSpeechToTextAdapter(SpeechToTextPort):
    name = "stub_stt"

    async def transcribe(self, audio_b64: str) -> str:
        return "stub transcription"
