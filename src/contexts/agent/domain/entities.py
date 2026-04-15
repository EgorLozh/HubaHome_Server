class AgentResponse:
    def __init__(self, text: str, confidence: float = 0.0) -> None:
        self.text = text
        self.confidence = confidence
