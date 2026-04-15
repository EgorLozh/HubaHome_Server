class SpeechChunk:
    def __init__(self, chunk_id: int, payload_b64: str) -> None:
        self.chunk_id = chunk_id
        self.payload_b64 = payload_b64
