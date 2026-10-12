"""Isolated CPU transformer inference API; requires an approved local model artifact."""
from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache
from threading import Lock

from fastapi import Depends, FastAPI, Header, HTTPException, Request
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


_requests: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def authorize(request: Request, x_api_key: str | None = Header(default=None)):
    """Fail closed; local deployment requires configured API key even for staging."""
    expected = os.getenv("TRANSFORMER_API_KEY", "")
    if len(expected) < 24 or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Single-process in-memory limiter. For multi-worker production use shared gateway storage.
    limit = int(os.getenv("TRANSFORMER_RPM_LIMIT", "30"))
    if limit < 1:
        raise HTTPException(status_code=503, detail="Rate limit configuration invalid")
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        q = _requests[client]
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


@app.post("/predict")
def predict(body: Request, _auth: None = Depends(authorize)):
    try:
        return service().predict(body.text)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Transformer inference unavailable") from exc
