from abc import ABC, abstractmethod
from datetime import datetime


class SchedulerPort(ABC):
    @abstractmethod
    async def schedule(self, task_id: str, run_at: datetime, payload: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, task_id: str) -> None:
        raise NotImplementedError
