from abc import ABC, abstractmethod


class SpeechToTextPort(ABC):
    @abstractmethod
    async def transcribe(self, audio_b64: str) -> str:
        raise NotImplementedError
