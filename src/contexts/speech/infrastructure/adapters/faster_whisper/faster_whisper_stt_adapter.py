import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any

from src.contexts.speech.application.ports.speech_to_text_port import SpeechToTextPort
from src.contexts.speech.infrastructure.adapters.faster_whisper.whisper_model_manager import (
    WhisperModelManager,
)


class FasterWhisperSpeechToTextAdapter(SpeechToTextPort):
    name = "faster_whisper"

    def __init__(self, model_manager: WhisperModelManager) -> None:
        self.model_manager = model_manager
        self._model: Any | None = None
        self.last_model_status: str = "not_initialized"

    def initialize(self) -> str:
        status, _ = self.model_manager.ensure_model()
        self._model = self.model_manager.load_model()
        self.last_model_status = status
        return status

    async def transcribe(self, audio_b64: str) -> str:
        if not audio_b64:
            return ""
        if self._model is None:
            await asyncio.to_thread(self.initialize)
        if self._model is None:
            raise RuntimeError("Whisper model failed to initialize")
        return await asyncio.to_thread(self._transcribe_sync, audio_b64)

    def _transcribe_sync(self, audio_b64: str) -> str:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = Path(tmp_file.name)

        try:
            segments, _info = self._model.transcribe(str(temp_path), vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text
        finally:
            temp_path.unlink(missing_ok=True)
