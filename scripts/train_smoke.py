"""Train a deterministic synthetic sentiment smoke baseline (not real accuracy)."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sentiment_intelligence.model import build_pipeline, save_model

POSITIVE = [
    "I love this product", "What a wonderful experience", "Excellent customer support",
    "The service is fantastic", "I am very happy", "Highly recommended purchase",
    "The quality is superb", "Everything worked beautifully", "Great value for money",
    "Amazing performance today", "The team was helpful", "I enjoyed using the app",
]
NEGATIVE = [
    "I hate this product", "What a terrible experience", "Awful customer support",
    "The service is disappointing", "I am very unhappy", "Do not recommend buying",
    "The quality is poor", "Nothing worked correctly", "Bad value for money",
    "Horrible performance today", "The team was unhelpful", "I disliked using the app",
]


def main() -> None:
    texts = POSITIVE + NEGATIVE
    labels = ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test), output_dict=True, zero_division=0)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    save_model(model, out / "sentiment_baseline.joblib")
    (out / "smoke_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Synthetic smoke baseline trained. Metrics are not real-world evidence.")


if __name__ == "__main__":
    main()
