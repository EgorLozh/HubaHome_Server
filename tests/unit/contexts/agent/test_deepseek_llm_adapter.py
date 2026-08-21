import pytest

from src.contexts.agent.infrastructure.adapters.deepseek.deepseek_llm_adapter import (
    DeepSeekLlmAdapter,
)


def test_deepseek_get_chat_model_requires_api_key():
    adapter = DeepSeekLlmAdapter(api_key="", base_url="https://api.deepseek.com", model="deepseek-chat")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        adapter.get_chat_model()


@pytest.mark.asyncio
async def test_deepseek_ping_returns_false_without_api_key():
    adapter = DeepSeekLlmAdapter(api_key="", base_url="https://api.deepseek.com", model="deepseek-chat")
    assert await adapter.ping() is False
