import pytest
from fastapi.testclient import TestClient

from src.bootstrap.app_factory import create_app
from src.shared.config.settings import Settings


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        api_key="test-api-key",
        qdrant_url="http://localhost:6333",
        ollama_url="http://localhost:11434",
        enable_real_stt=False,
        enable_real_tts=False,
        enable_real_web_search=False,
        knowledge_metadata_path=str(tmp_path / "metadata.json"),
    )


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = create_app(settings=settings)
    return TestClient(app)
