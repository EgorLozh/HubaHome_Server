import argparse
import asyncio
import base64
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contexts.speech.infrastructure.adapters.faster_whisper.faster_whisper_stt_adapter import (
    FasterWhisperSpeechToTextAdapter,
)
from src.contexts.speech.infrastructure.adapters.faster_whisper.whisper_model_manager import (
    WhisperModelManager,
)
from src.shared.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check for Whisper STT")
    parser.add_argument(
        "--audio",
        default=".tmp/piper_smoke.wav",
        help="Path to WAV audio file (default: .tmp/piper_smoke.wav)",
    )
    parser.add_argument(
        "--expect-contains",
        default="",
        help="Optional substring expected in transcription",
    )
    return parser.parse_args()


async def run(audio_path: str, expect_contains: str) -> int:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise RuntimeError("Only WAV input is supported for this smoke check")

    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    if not audio_b64:
        raise RuntimeError("Input audio is empty")

    settings = get_settings()
    manager = WhisperModelManager(
        model_name=settings.whisper_model_name,
        models_dir=settings.whisper_models_dir,
        auto_download=settings.whisper_auto_download,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    adapter = FasterWhisperSpeechToTextAdapter(model_manager=manager)

    model_status = adapter.initialize()
    text = await adapter.transcribe(audio_b64)
    if not text.strip():
        raise RuntimeError("Whisper returned empty transcription")

    print(f"[ok] model: {settings.whisper_model_name}")
    print(f"[ok] model status: {model_status}")
    print(f"[ok] audio: {path}")
    print(f"[ok] transcription: {text}")

    if expect_contains and expect_contains.lower() not in text.lower():
        raise RuntimeError(
            f"Transcription does not include expected substring: '{expect_contains}'"
        )

    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args.audio, args.expect_contains))
    except Exception as err:  # pragma: no cover - smoke script
        print(f"[fail] whisper smoke check failed: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
