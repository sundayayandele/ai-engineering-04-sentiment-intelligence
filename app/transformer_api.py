"""Isolated CPU transformer inference API; requires an approved local model artifact."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_intelligence.transformer_service import TransformerService

app = FastAPI(title="Project 04 Transformer Staging API", version="0.2.0")


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@lru_cache(maxsize=1)
def service():
    return TransformerService(os.environ.get("TRANSFORMER_MODEL_DIR", "/app/model"))


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    try:
        service()
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Transformer artifact unavailable") from exc
    return {"status": "ready"}


@app.post("/predict")
def predict(body: Request):
    try:
        return service().predict(body.text)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Transformer inference unavailable") from exc
