def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_has_no_secrets(client):
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert body["apiKeyConfigured"] is True
    assert "api_key" not in body
