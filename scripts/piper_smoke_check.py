import argparse
import asyncio
import base64
import io
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contexts.speech.infrastructure.adapters.piper.piper_model_manager import (
    PiperModelManager,
)
from src.contexts.speech.infrastructure.adapters.piper.piper_text_to_speech_adapter import (
    PiperTextToSpeechAdapter,
)
from src.shared.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check for Piper TTS")
    parser.add_argument(
        "--text",
        default="Привет! Это тест синтеза речи через Piper.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output WAV path",
    )
    return parser.parse_args()


def inspect_wav(wav_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.getnframes()
        if frames <= 0:
            raise RuntimeError("WAV has no audio frames")
        return sample_rate, channels, frames


async def run(text: str, output_path: str) -> int:
    settings = get_settings()
    manager = PiperModelManager(
        model_name=settings.piper_model_name,
        models_dir=settings.piper_models_dir,
        auto_download=settings.piper_auto_download,
    )
    adapter = PiperTextToSpeechAdapter(
        model_manager=manager,
        speaker_id=settings.piper_speaker_id,
        use_cuda=settings.piper_use_cuda,
    )

    model_status = adapter.initialize()
    audio_base64 = await adapter.synthesize(text)
    if not audio_base64:
        raise RuntimeError("Piper returned empty audio payload")

    wav_bytes = base64.b64decode(audio_base64)
    sample_rate, channels, frames = inspect_wav(wav_bytes)

    print(f"[ok] model: {settings.piper_model_name}")
    print(f"[ok] model status: {model_status}")
    print(f"[ok] audio: {len(wav_bytes)} bytes, {sample_rate} Hz, {channels} ch, {frames} frames")

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(wav_bytes)
        print(f"[ok] saved wav: {output}")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args.text, args.output))
    except Exception as err:  # pragma: no cover - smoke script
        print(f"[fail] piper smoke check failed: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
import argparse
import asyncio
import base64
import shutil
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contexts.speech.infrastructure.adapters.piper.piper_text_to_speech_adapter import (
    PiperTextToSpeechAdapter,
)
from src.shared.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test for Piper TTS adapter."
    )
    parser.add_argument(
        "--text",
        default="Piper smoke test from HubaHome",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="tmp/piper_smoke_test.wav",
        help="Path to output WAV file.",
    )
    parser.add_argument(
        "--bin-path",
        default=None,
        help="Override Piper executable path. Default: PIPER_BIN_PATH.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override Piper model path. Default: PIPER_MODEL_PATH.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="Override speaker id. Default: PIPER_SPEAKER_ID.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate for WAV wrapper (raw output metadata is not embedded).",
    )
    return parser.parse_args()


def write_wav(path: Path, raw_pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_pcm)


async def run_smoke_test(args: argparse.Namespace) -> int:
    settings = get_settings()

    bin_path = args.bin_path or settings.piper_bin_path
    model_path = args.model_path or settings.piper_model_path
    speaker_id = (
        args.speaker_id
        if args.speaker_id is not None
        else settings.piper_speaker_id
    )

    if not model_path:
        print("[fail] PIPER_MODEL_PATH is empty. Set it in .env or via --model-path.")
        return 1

    if not Path(model_path).exists():
        print(f"[fail] Piper model file not found: {model_path}")
        return 1

    # `PIPER_BIN_PATH` can be either an absolute/relative path or a command from PATH.
    bin_path_obj = Path(bin_path)
    if bin_path_obj.exists() and bin_path_obj.is_file():
        resolved_bin_path = str(bin_path_obj.resolve())
    else:
        resolved_bin_path = shutil.which(bin_path)

    if not resolved_bin_path:
        print(
            "[fail] Piper executable not found. "
            f"Set PIPER_BIN_PATH to a valid file path or install 'piper' into PATH. "
            f"Current value: {bin_path}"
        )
        return 1

    adapter = PiperTextToSpeechAdapter(
        bin_path=resolved_bin_path,
        model_path=model_path,
        speaker_id=speaker_id,
    )

    try:
        audio_base64 = await adapter.synthesize(args.text)
    except Exception as err:
        print(f"[fail] Piper synthesis failed: {err}")
        return 1

    if not audio_base64:
        print("[fail] Empty audio returned by Piper.")
        return 1

    raw_pcm = base64.b64decode(audio_base64)
    output_path = Path(args.output)
    write_wav(output_path, raw_pcm, sample_rate=args.sample_rate)

    print("[ok] Piper smoke test passed")
    print(f"[ok] Bin: {resolved_bin_path}")
    print(f"[ok] Model: {model_path}")
    print(f"[ok] Speaker: {speaker_id}")
    print(f"[ok] Output: {output_path.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_smoke_test(args))


if __name__ == "__main__":
    sys.exit(main())
