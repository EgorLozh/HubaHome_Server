import base64
import io
import wave

import pytest
from starlette.websockets import WebSocketDisconnect

from src.contexts.conversation.application.use_cases.orchestrate_turn_agent import TurnOutput


def test_ws_requires_api_key(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/voice/stream"):
            pass


def test_ws_accepts_mvp_event_set(client):
    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
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
    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "final_transcript", "text": "какая погода"})
        payload = websocket.receive_json()
        assert payload["event"] == "error"
        assert "Wakeword" in payload["message"]


def test_ws_prefers_server_stt_when_audio_is_present(client, monkeypatch):
    captured: dict[str, str] = {}

    async def fake_transcribe(audio_b64: str) -> str:
        assert audio_b64 == "UklGRg=="
        return "серверный текст"

    async def fake_execute(transcript: str) -> TurnOutput:
        captured["transcript"] = transcript
        return TurnOutput(assistant_text="ok", intent="chat")

    async def fake_synthesize(_: str) -> str:
        return ""

    monkeypatch.setattr(client.app.state.container.speech_to_text_adapter, "name", "fake_stt")
    monkeypatch.setattr(client.app.state.container.speech_to_text_adapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(client.app.state.container.orchestrate_turn_use_case, "execute", fake_execute)
    monkeypatch.setattr(client.app.state.container.text_to_speech_adapter, "synthesize", fake_synthesize)

    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        websocket.send_json({"event": "audio_chunk", "chunkId": 0, "payloadB64": "UklGRg=="})
        websocket.send_json({"event": "final_transcript", "text": "клиентский текст"})

        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"
        assert payload["text"] == "ok"
        assert captured["transcript"] == "серверный текст"


def test_ws_merges_streamed_audio_chunks_in_order(client, monkeypatch):
    captured: dict[str, bytes | str] = {}

    async def fake_transcribe(audio_b64: str) -> str:
        wav_bytes = base64.b64decode(audio_b64)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            captured["pcm"] = wav_file.readframes(wav_file.getnframes())
        return "склеенный текст"

    async def fake_execute(transcript: str) -> TurnOutput:
        captured["transcript"] = transcript
        return TurnOutput(assistant_text="ok", intent="chat")

    async def fake_synthesize(_: str) -> str:
        return ""

    monkeypatch.setattr(client.app.state.container.speech_to_text_adapter, "name", "fake_stt")
    monkeypatch.setattr(client.app.state.container.speech_to_text_adapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(client.app.state.container.orchestrate_turn_use_case, "execute", fake_execute)
    monkeypatch.setattr(client.app.state.container.text_to_speech_adapter, "synthesize", fake_synthesize)

    chunk_one = _pcm_chunk_to_wav_b64(b"\x01\x00\x02\x00")
    chunk_two = _pcm_chunk_to_wav_b64(b"\x03\x00\x04\x00")

    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        websocket.send_json({"event": "audio_chunk", "chunkId": 1, "payloadB64": chunk_two})
        websocket.send_json({"event": "audio_chunk", "chunkId": 0, "payloadB64": chunk_one})
        websocket.send_json({"event": "final_transcript", "text": ""})

        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"
        assert payload["text"] == "ok"
        assert captured["transcript"] == "склеенный текст"
        assert captured["pcm"] == b"\x01\x00\x02\x00\x03\x00\x04\x00"


def test_ws_falls_back_to_client_text_when_stub_stt_is_active(client, monkeypatch):
    captured: dict[str, str] = {}

    async def fake_execute(transcript: str) -> TurnOutput:
        captured["transcript"] = transcript
        return TurnOutput(assistant_text="ok", intent="chat")

    async def fake_synthesize(_: str) -> str:
        return ""

    monkeypatch.setattr(client.app.state.container.orchestrate_turn_use_case, "execute", fake_execute)
    monkeypatch.setattr(client.app.state.container.text_to_speech_adapter, "synthesize", fake_synthesize)

    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        websocket.send_json({"event": "audio_chunk", "chunkId": 0, "payloadB64": "UklGRg=="})
        websocket.send_json({"event": "final_transcript", "text": "клиентский fallback"})

        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"
        assert payload["text"] == "ok"
        assert captured["transcript"] == "клиентский fallback"


def test_ws_metadata_and_document_commands_return_assistant_text(client):
    with client.websocket_connect("/v1/voice/stream?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
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


def test_legacy_voice_session_alias_still_works(client):
    with client.websocket_connect("/v1/voice/session?api_key=test-api-key") as websocket:
        websocket.send_json({"event": "wakeword_detected"})
        websocket.send_json({"event": "partial_transcript", "text": "ping"})
        payload = websocket.receive_json()
        assert payload["event"] == "assistant_text"


def _pcm_chunk_to_wav_b64(pcm: bytes) -> str:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm)
    return base64.b64encode(wav_buffer.getvalue()).decode("ascii")
