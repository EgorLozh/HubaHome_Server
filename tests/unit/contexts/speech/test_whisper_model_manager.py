from pathlib import Path

import pytest

from src.contexts.speech.infrastructure.adapters.faster_whisper.whisper_model_manager import (
    WhisperModelManager,
)


def test_whisper_model_manager_reports_ready_when_files_exist(tmp_path: Path):
    (tmp_path / "small-model-marker").write_text("ready")

    manager = WhisperModelManager(
        model_name="small",
        models_dir=str(tmp_path),
        auto_download=True,
    )
    status, directory = manager.ensure_model()

    assert status == "ready"
    assert directory == str(tmp_path)


def test_whisper_model_manager_downloads_when_missing(tmp_path: Path, monkeypatch):
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            (tmp_path / "small-download-marker").write_text("downloaded")

    def fake_import_module(_name: str):
        class FakeModule:
            WhisperModel = FakeWhisperModel

        return FakeModule()

    monkeypatch.setattr(
        "src.contexts.speech.infrastructure.adapters.faster_whisper.whisper_model_manager.importlib.import_module",
        fake_import_module,
    )

    manager = WhisperModelManager(
        model_name="small",
        models_dir=str(tmp_path),
        auto_download=True,
    )
    status, _ = manager.ensure_model()
    assert status == "downloaded"


def test_whisper_model_manager_raises_if_download_disabled(tmp_path: Path):
    manager = WhisperModelManager(
        model_name="small",
        models_dir=str(tmp_path),
        auto_download=False,
    )
    with pytest.raises(RuntimeError):
        manager.ensure_model()
