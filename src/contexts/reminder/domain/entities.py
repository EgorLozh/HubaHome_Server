from datetime import datetime


class Reminder:
    def __init__(self, reminder_id: str, text: str, due_at: datetime) -> None:
        self.reminder_id = reminder_id
        self.text = text
        self.due_at = due_at
