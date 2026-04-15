def present_text(text: str) -> dict[str, str]:
    return {"event": "assistant_text", "text": text}
