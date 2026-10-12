"""Sentiment API; no model downloaded or external requests at startup."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_intelligence.model import load_model

app = FastAPI(title="Project 04 Sentiment Intelligence", version="0.1.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    model_type: str = "synthetic-smoke-tfidf-logistic-regression"


@lru_cache
def get_model():
    path = Path(os.getenv("MODEL_PATH", "artifacts/sentiment_baseline.joblib"))
    if not path.is_file():
        raise RuntimeError("Model artifact unavailable: train the smoke baseline first")
    return load_model(path)


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    try:
        get_model()
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Model unavailable") from exc
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    try:
        model = get_model()
        probabilities = model.predict_proba([body.text])[0]
        by_class = dict(zip(model.classes_.tolist(), probabilities.tolist(), strict=True))
        sentiment = max(by_class, key=by_class.get)
        return PredictionResponse(
            sentiment=sentiment, confidence=by_class[sentiment], probabilities=by_class
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Prediction unavailable") from exc
