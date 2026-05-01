from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProviderPort(ABC):
    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        raise NotImplementedError

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        raise NotImplementedError
