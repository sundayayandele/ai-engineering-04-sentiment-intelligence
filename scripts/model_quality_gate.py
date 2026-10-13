"""Auditable staging-only sentiment quality gates; never implies production approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(report: dict, *, min_macro_f1: float = 0.40) -> dict:
    labels = ("negative", "neutral", "positive")
    model = report.get("transformer", {})
    baseline = report.get("baseline", {})
    failures = []
    if not model:
        failures.append("transformer_metrics_missing")
    else:
        matrix = model.get("confusion_matrix", [])
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            failures.append("invalid_confusion_matrix")
        else:
            predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
            if any(n == 0 for n in predicted):
                failures.append("class_collapse")
        if model.get("macro_f1", 0) < min_macro_f1:
            failures.append("macro_f1_below_threshold")
        classification = model.get("classification", {})
        for label in labels:
            if classification.get(label, {}).get("recall", 0) <= 0:
                failures.append(f"zero_recall_{label}")
        if model.get("macro_f1", 0) < baseline.get("macro_f1", 0):
            failures.append("worse_than_tfidf_baseline")
    return {
        "status": "pass" if not failures else "fail",
        "stage": "experimental-quality-only",
        "production_approved": False,
        "min_macro_f1": min_macro_f1,
        "model_macro_f1": model.get("macro_f1"),
        "baseline_macro_f1": baseline.get("macro_f1"),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--min-macro-f1", type=float, default=0.40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.comparison.read_text()), min_macro_f1=args.min_macro_f1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
