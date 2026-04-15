from fastapi import WebSocket


def resolve_api_key(websocket: WebSocket) -> str:
    header_key = websocket.headers.get("x-api-key")
    query_key = websocket.query_params.get("api_key")
    return header_key or query_key or ""


def is_authorized(provided_key: str, expected_key: str) -> bool:
    return bool(provided_key) and provided_key == expected_key
