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


def test_ws_accepts_single_utterance_audio_for_server_stt(client):
    with client.websocket_connect("/v1/voice/session?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        assert websocket.receive_json()["event"] == "assistant_text"

        websocket.send_json({"event": "audio_chunk", "chunkId": 0, "payloadB64": "UklGRg=="})
        websocket.send_json({"event": "final_transcript", "text": ""})

        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"
        assert payload["text"]


def test_ws_metadata_and_document_commands_return_assistant_text(client):
    with client.websocket_connect("/v1/voice/session?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        assert websocket.receive_json()["event"] == "assistant_text"

        websocket.send_json({"event": "final_transcript", "text": "меня зовут Егор"})
        set_name_payload = websocket.receive_json()
        assert set_name_payload["event"] == "assistant_text"
        assert set_name_payload["text"]

        websocket.send_json({"event": "final_transcript", "text": "запомни документ про молоко"})
        save_payload = websocket.receive_json()
        assert save_payload["event"] == "assistant_text"
        assert save_payload["text"]

        websocket.send_json({"event": "final_transcript", "text": "что ты знаешь про молоко"})
        search_payload = websocket.receive_json()
        assert search_payload["event"] == "assistant_text"
        assert search_payload["text"]
