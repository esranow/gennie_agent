import os
import requests
from fastapi.testclient import TestClient
from main import app
from ingestion import build_index, collection
import routing

def test_pipeline():
    if collection.count() == 0:
        build_index()
    assert collection.count() > 0
    
    client = TestClient(app)
    resp = client.post("/query", json={"query": "What is the attention mechanism?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "mode" in data
    assert "think" in data
    assert "final" in data
    assert "confidence" in data
    assert "backend" in data
    assert "latency_ms" in data
    
    orig_head = requests.head
    def mock_head(*args, **kwargs):
        raise requests.ConnectionError("Mocked offline")
    requests.head = mock_head
    try:
        assert routing.get_backend() == "local"
    finally:
        requests.head = orig_head
        
    print("All tests passed.")

if __name__ == "__main__":
    test_pipeline()
