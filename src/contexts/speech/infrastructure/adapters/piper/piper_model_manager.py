from pathlib import Path

from piper.download_voices import download_voice


class PiperModelManager:
    def __init__(self, model_name: str, models_dir: str, auto_download: bool = True) -> None:
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.auto_download = auto_download

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.model_name}.onnx"

    @property
    def config_path(self) -> Path:
        return self.models_dir / f"{self.model_name}.onnx.json"

    def is_model_ready(self) -> bool:
        return self.model_path.exists() and self.config_path.exists()

    def ensure_model(self) -> tuple[Path, Path, str]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.is_model_ready():
            return self.model_path, self.config_path, "ready"

        if not self.auto_download:
            raise RuntimeError(
                f"Piper model '{self.model_name}' is missing and auto-download is disabled"
            )

        download_voice(self.model_name, self.models_dir)
        if not self.is_model_ready():
            raise RuntimeError(f"Failed to download Piper model '{self.model_name}'")
        return self.model_path, self.config_path, "downloaded"
