def test_diagnostics_endpoint(client):
    response = client.get("/diagnostics")
    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    assert "qdrant" in body["checks"]
    assert "ollama" in body["checks"]
