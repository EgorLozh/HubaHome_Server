import importlib
from pathlib import Path
from typing import Any


class WhisperModelManager:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        auto_download: bool = True,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.auto_download = auto_download
        self.device = device
        self.compute_type = compute_type

    def _load_whisper_class(self):
        module = importlib.import_module("faster_whisper")
        return getattr(module, "WhisperModel")

    def _has_local_model_files(self) -> bool:
        if not self.models_dir.exists():
            return False
        matcher = f"**/*{self.model_name}*"
        return any(self.models_dir.glob(matcher))

    def ensure_model(self) -> tuple[str, str]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self._has_local_model_files():
            return "ready", str(self.models_dir)

        if not self.auto_download:
            raise RuntimeError(
                f"Whisper model '{self.model_name}' is missing and auto-download is disabled"
            )

        whisper_cls = self._load_whisper_class()
        whisper_cls(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.models_dir),
            local_files_only=False,
        )
        return "downloaded", str(self.models_dir)

    def load_model(self) -> Any:
        whisper_cls = self._load_whisper_class()
        return whisper_cls(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.models_dir),
            local_files_only=not self.auto_download,
        )
