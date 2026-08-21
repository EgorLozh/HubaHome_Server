def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_has_no_secrets(client):
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert body["apiKeyConfigured"] is True
    assert body["llmProvider"] == "ollama"
    assert body["deepseekApiKeyConfigured"] is False
    assert "api_key" not in body
    assert "deepseek_api_key" not in body
    assert "deepseekApiKey" not in body
