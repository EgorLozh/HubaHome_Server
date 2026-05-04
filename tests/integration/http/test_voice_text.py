from src.interfaces.voice.turn_runtime import VoiceTurnResult


def test_voice_text_endpoint_requires_api_key(client):
    response = client.post("/v1/voice/text", json={"text": "привет"})
    assert response.status_code == 401


def test_voice_text_endpoint_returns_turn_result(client, monkeypatch):
    async def fake_run_voice_turn(container, transcript: str):  # noqa: ARG001
        assert transcript == "какая погода"
        return VoiceTurnResult(
            assistant_text="Солнечно",
            assistant_audio_b64="YmFzZTY0",
            intent="weather",
        )

    monkeypatch.setattr("src.interfaces.http.voice_text_router.run_voice_turn", fake_run_voice_turn)

    response = client.post(
        "/v1/voice/text",
        headers={"x-api-key": "test-api-key"},
        json={"text": "какая погода", "correlationId": "corr-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "assistantText": "Солнечно",
        "assistantAudioB64": "YmFzZTY0",
        "intent": "weather",
        "correlationId": "corr-1",
    }
