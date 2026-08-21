from src.bootstrap.wiring import build_container
from src.shared.config.settings import Settings


def test_build_container_returns_stubbed_adapters():
    settings = Settings(
        api_key="test-api-key",
        llm_provider="ollama",
        enable_real_stt=False,
        enable_real_tts=False,
    )
    container = build_container(settings)

    assert container.settings.api_key == "test-api-key"
    assert container.llm_adapter.name == "ollama"
    assert container.vector_store_adapter.name == "qdrant"
    assert container.speech_to_text_adapter.name == "stub_stt"
    assert container.text_to_speech_adapter.name == "stub_tts"
    assert container.web_search_adapter.name == "duckduckgo_web_search"
    assert container.notification_adapter.name == "stub_notification"
    assert container.orchestrate_turn_use_case is not None


def test_build_container_uses_deepseek_adapter_when_configured():
    settings = Settings(
        api_key="test-api-key",
        llm_provider="deepseek",
        deepseek_api_key="sk-test",
        enable_real_stt=False,
        enable_real_tts=False,
    )
    container = build_container(settings)
    assert container.llm_adapter.name == "deepseek"


def test_build_container_rejects_unknown_llm_provider():
    settings = Settings(
        api_key="test-api-key",
        llm_provider="unknown",
        enable_real_stt=False,
        enable_real_tts=False,
    )
    try:
        build_container(settings)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported LLM_PROVIDER" in str(exc)


def test_build_container_can_enable_real_whisper_adapter():
    settings = Settings(
        api_key="test-api-key",
        enable_real_stt=True,
        whisper_auto_download=False,
    )
    container = build_container(settings)
    assert container.speech_to_text_adapter.name == "faster_whisper"
    assert container.whisper_model_status == "pending"
