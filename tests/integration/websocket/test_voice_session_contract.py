import pytest
from starlette.websockets import WebSocketDisconnect


def test_ws_requires_api_key(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/voice/session"):
            pass


def test_ws_accepts_mvp_event_set(client):
    with client.websocket_connect("/v1/voice/session?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        assert websocket.receive_json()["event"] == "assistant_text"

        websocket.send_json({"event": "audio_chunk", "chunkId": 1, "payloadB64": ""})

        websocket.send_json({"event": "partial_transcript", "text": "какая"})
        assert websocket.receive_json()["event"] == "assistant_text"

        websocket.send_json({"event": "final_transcript", "text": "какая погода"})
        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"
        assert payload["text"]

        websocket.send_json({"event": "assistant_text", "text": "hello"})
        assert websocket.receive_json()["event"] == "assistant_text"

        websocket.send_json({"event": "assistant_audio_chunk", "chunkId": 2, "payloadB64": ""})
        assert websocket.receive_json()["event"] == "assistant_audio_chunk"

        websocket.send_json({"event": "error", "message": "manual"})
        assert websocket.receive_json()["event"] == "error"


def test_ws_rejects_final_transcript_without_wakeword(client):
    with client.websocket_connect("/v1/voice/session?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "final_transcript", "text": "какая погода"})
        payload = websocket.receive_json()
        assert payload["event"] == "error"
        assert "Wakeword" in payload["message"]
