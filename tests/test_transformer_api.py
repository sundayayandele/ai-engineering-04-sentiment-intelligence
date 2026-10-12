from fastapi.testclient import TestClient
import app.transformer_api as module


def test_live_and_not_ready(monkeypatch):
    module.service.cache_clear()
    monkeypatch.setenv("TRANSFORMER_MODEL_DIR", "/not/a/real/model")
    client = TestClient(module.app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.post("/predict", json={"text": ""}).status_code == 422
    module.service.cache_clear()


def test_mocked_transformer_prediction(monkeypatch):
    class Stub:
        def predict(self, text):
            return {"sentiment": "positive", "confidence": 0.9,
                    "probabilities": {"negative": 0.05, "neutral": 0.05,
                                      "positive": 0.9},
                    "model_type": "approved-local-transformer"}
    monkeypatch.setattr(module, "service", lambda: Stub())
    client = TestClient(module.app)
    assert client.get("/health/ready").status_code == 200
    assert client.post("/predict", json={"text": "great"}).json()["sentiment"] == "positive"
