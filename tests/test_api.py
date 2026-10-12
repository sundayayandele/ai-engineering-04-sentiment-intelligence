from fastapi.testclient import TestClient

from app.main import app, get_model
from sentiment_intelligence.model import build_pipeline


def test_live():
    assert TestClient(app).get("/health/live").status_code == 200


def test_prediction():
    model = build_pipeline()
    model.fit(["great product", "lovely product", "awful product", "terrible product"],
              ["positive", "positive", "negative", "negative"])
    app.dependency_overrides.clear()
    get_model.cache_clear()
    # The API's cached loader can be replaced for this unit test without a model artifact.
    import app.main as module
    original = module.get_model
    module.get_model = lambda: model
    try:
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200
            response = client.post("/predict", json={"text": "great product"})
            assert response.status_code == 200
            body = response.json()
            assert body["sentiment"] in ("positive", "negative")
            assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
            assert client.post("/predict", json={"text": ""}).status_code == 422
    finally:
        module.get_model = original
