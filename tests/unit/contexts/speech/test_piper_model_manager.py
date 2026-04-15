from pathlib import Path

import pytest

from src.contexts.speech.infrastructure.adapters.piper.piper_model_manager import (
    PiperModelManager,
)


def test_piper_model_manager_detects_existing_model(tmp_path: Path):
    model_name = "ru_RU-irina-medium"
    (tmp_path / f"{model_name}.onnx").write_bytes(b"onnx")
    (tmp_path / f"{model_name}.onnx.json").write_text("{}")

    manager = PiperModelManager(model_name=model_name, models_dir=str(tmp_path), auto_download=True)
    model_path, config_path, status = manager.ensure_model()

    assert status == "ready"
    assert model_path.exists()
    assert config_path.exists()


def test_piper_model_manager_downloads_missing_model(tmp_path: Path, monkeypatch):
    model_name = "ru_RU-irina-medium"

    def fake_download(voice: str, download_dir: Path, force_redownload: bool = False) -> None:
        assert voice == model_name
        (download_dir / f"{voice}.onnx").write_bytes(b"onnx")
        (download_dir / f"{voice}.onnx.json").write_text("{}")

    monkeypatch.setattr(
        "src.contexts.speech.infrastructure.adapters.piper.piper_model_manager.download_voice",
        fake_download,
    )

    manager = PiperModelManager(model_name=model_name, models_dir=str(tmp_path), auto_download=True)
    _, _, status = manager.ensure_model()
    assert status == "downloaded"


def test_piper_model_manager_raises_when_auto_download_disabled(tmp_path: Path):
    manager = PiperModelManager(
        model_name="ru_RU-irina-medium",
        models_dir=str(tmp_path),
        auto_download=False,
    )
    with pytest.raises(RuntimeError):
        manager.ensure_model()
