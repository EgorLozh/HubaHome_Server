from src.contexts.notification.application.ports.notification_port import NotificationPort


class StubNotificationAdapter(NotificationPort):
    name = "stub_notification"

    async def send(self, text: str, reminder_id: str | None = None, scheduled_at: str | None = None) -> None:
        return None
