from src.bootstrap.wiring import build_container
from src.shared.config.settings import Settings


def test_build_container_returns_stubbed_adapters():
    settings = Settings(api_key="test-api-key")
    container = build_container(settings)

    assert container.settings.api_key == "test-api-key"
    assert container.llm_adapter.name == "ollama"
    assert container.vector_store_adapter.name == "qdrant"
    assert container.notification_adapter.name == "stub_notification"
