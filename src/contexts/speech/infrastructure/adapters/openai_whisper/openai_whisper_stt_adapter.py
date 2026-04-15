import base64

import httpx

from src.contexts.speech.application.ports.speech_to_text_port import SpeechToTextPort


class OpenAiWhisperSpeechToTextAdapter(SpeechToTextPort):
    name = "openai_whisper"

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def transcribe(self, audio_b64: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if not audio_b64:
            return ""

        audio_bytes = base64.b64decode(audio_b64)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            payload = response.json()
        return payload.get("text", "").strip()
