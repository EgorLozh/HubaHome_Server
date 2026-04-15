class NotificationMessage:
    def __init__(self, text: str, reminder_id: str | None = None) -> None:
        self.text = text
        self.reminder_id = reminder_id
