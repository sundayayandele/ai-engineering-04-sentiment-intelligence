"""Auditable staging-only sentiment quality gates; never implies production approval."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def assess(report: dict, *, min_macro_f1: float = 0.40) -> dict:
    labels = ("negative", "neutral", "positive")
    model = report.get("transformer", {})
    baseline = report.get("baseline", {})
    failures = []
    for name, metrics in (("transformer", model), ("baseline", baseline)):
        score = metrics.get("macro_f1")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            failures.append(f"invalid_{name}_macro_f1")
    if not model:
        failures.append("transformer_metrics_missing")
    else:
        matrix = model.get("confusion_matrix", [])
        if (not isinstance(matrix, list) or len(matrix) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                       for row in matrix for v in row)):
            failures.append("invalid_confusion_matrix")
        else:
            predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
            if sum(predicted) == 0:
                failures.append("empty_confusion_matrix")
            if any(n == 0 for n in predicted):
                failures.append("class_collapse")
        if isinstance(model.get("macro_f1"), (int, float)) and math.isfinite(model["macro_f1"]) and model["macro_f1"] < min_macro_f1:
            failures.append("macro_f1_below_threshold")
        classification = model.get("classification", {})
        for label in labels:
            if classification.get(label, {}).get("recall", 0) <= 0:
                failures.append(f"zero_recall_{label}")
        if (isinstance(model.get("macro_f1"), (int, float))
                and isinstance(baseline.get("macro_f1"), (int, float))
                and math.isfinite(model["macro_f1"])
                and math.isfinite(baseline["macro_f1"])
                and model["macro_f1"] < baseline["macro_f1"]):
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
