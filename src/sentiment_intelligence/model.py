"""CPU-friendly TF-IDF sentiment baseline. Not a validated production model."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ])


def save_model(model: Pipeline, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> Pipeline:
    # Never load joblib files from untrusted sources: deserialization can execute code.
    return joblib.load(path)
