import base64
import io
import wave

from piper.voice import PiperVoice, SynthesisConfig

from src.contexts.speech.application.ports.text_to_speech_port import TextToSpeechPort
from src.contexts.speech.infrastructure.adapters.piper.piper_model_manager import (
    PiperModelManager,
)


class PiperTextToSpeechAdapter(TextToSpeechPort):
    name = "piper_python"

    def __init__(self, model_manager: PiperModelManager, speaker_id: int = 0, use_cuda: bool = False) -> None:
        self.model_manager = model_manager
        self.speaker_id = speaker_id
        self.use_cuda = use_cuda
        self._voice: PiperVoice | None = None
        self.last_model_status: str = "not_initialized"

    def initialize(self) -> str:
        model_path, config_path, status = self.model_manager.ensure_model()
        self._voice = PiperVoice.load(
            model_path=model_path,
            config_path=config_path,
            use_cuda=self.use_cuda,
            download_dir=self.model_manager.models_dir,
        )
        self.last_model_status = status
        return status

    async def synthesize(self, text: str) -> str:
        if not text.strip():
            return ""

        if self._voice is None:
            self.initialize()
        if self._voice is None:
            raise RuntimeError("Piper voice failed to initialize")

        synth_config = SynthesisConfig(speaker_id=self.speaker_id)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=synth_config)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
