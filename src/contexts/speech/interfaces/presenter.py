def present_audio_chunk(chunk_id: int, payload_b64: str) -> dict[str, object]:
    return {"event": "assistant_audio_chunk", "chunkId": chunk_id, "payloadB64": payload_b64}
