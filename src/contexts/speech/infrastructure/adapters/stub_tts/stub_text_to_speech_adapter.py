from src.contexts.speech.application.ports.text_to_speech_port import TextToSpeechPort


class StubTextToSpeechAdapter(TextToSpeechPort):
    name = "stub_tts"

    async def synthesize(self, text: str) -> str:
        return ""
