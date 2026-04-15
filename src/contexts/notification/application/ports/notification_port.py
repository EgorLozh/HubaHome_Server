from abc import ABC, abstractmethod


class NotificationPort(ABC):
    @abstractmethod
    async def send(self, text: str, reminder_id: str | None = None, scheduled_at: str | None = None) -> None:
        raise NotImplementedError
