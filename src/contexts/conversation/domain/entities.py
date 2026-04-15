class VoiceCommand:
    def __init__(self, text: str, language: str = "ru") -> None:
        self.text = text
        self.language = language
