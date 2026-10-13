"""Calibration diagnostics computed only from held-out validation logits."""
from __future__ import annotations

import numpy as np


def calibration_report(logits, labels_true, labels_train=None, bins: int = 10) -> dict:
    """Report uncalibrated confidence quality; no test-set parameter fitting."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels_true, dtype=int)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[0] != len(truth):
        raise ValueError("Expected nonempty [samples, classes] logits and matching labels")
    if not np.isfinite(scores).all() or not np.isin(truth, np.arange(scores.shape[1])).all():
        raise ValueError("Invalid logits or class labels")
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(float)
    onehot = np.eye(scores.shape[1])[truth]
    ece = 0.0
    histogram = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (confidence >= lo) & (confidence <= hi if i == bins - 1 else confidence < hi)
        n = int(mask.sum())
        if n:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            ece += n / len(truth) * abs(accuracy - mean_confidence)
            histogram.append({"lower": lo, "upper": hi, "count": n,
                              "accuracy": accuracy, "confidence": mean_confidence})
    return {
        "samples": len(truth),
        "ece": float(ece),
        "brier_score": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "negative_log_likelihood": float(-np.log(np.maximum(probs[np.arange(len(truth)), truth], 1e-12)).mean()),
        "bins": histogram,
        "split": "validation",
        "calibration_fitted": False,
    }
