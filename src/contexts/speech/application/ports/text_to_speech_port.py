from abc import ABC, abstractmethod


class TextToSpeechPort(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> str:
        raise NotImplementedError
