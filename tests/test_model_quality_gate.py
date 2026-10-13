"""Fast deterministic checks for experimental sentiment promotion policy."""
from scripts.model_quality_gate import assess


def report(predicted=(3, 3, 3), macro_f1=0.65, baseline=0.50):
    return {
        "baseline": {"macro_f1": baseline},
        "transformer": {
            "macro_f1": macro_f1,
            "confusion_matrix": [[predicted[i] if i == j else 0 for j in range(3)] for i in range(3)],
            "classification": {
                label: {"recall": 1.0 if predicted[i] else 0.0}
                for i, label in enumerate(("negative", "neutral", "positive"))
            },
        },
    }


def test_gate_accepts_qualified_experimental_model():
    result = assess(report())
    assert result["status"] == "pass"
    assert result["production_approved"] is False


def test_gate_rejects_class_collapse():
    result = assess(report(predicted=(0, 0, 9)))
    assert "class_collapse" in result["failures"]
    assert "zero_recall_negative" in result["failures"]


def test_gate_rejects_low_quality_and_regression():
    result = assess(report(macro_f1=0.2, baseline=0.5))
    assert "macro_f1_below_threshold" in result["failures"]
    assert "worse_than_tfidf_baseline" in result["failures"]


def test_gate_rejects_missing_model():
    assert "transformer_metrics_missing" in assess({"baseline": {}})["failures"]


def test_gate_rejects_missing_baseline():
    assert "invalid_baseline_macro_f1" in assess({"transformer": report()["transformer"]})["failures"]


def test_gate_rejects_nan_scores():
    data = report()
    data["transformer"]["macro_f1"] = float("nan")
    assert "invalid_transformer_macro_f1" in assess(data)["failures"]


def test_gate_rejects_empty_confusion_matrix():
    data = report()
    data["transformer"]["confusion_matrix"] = [[0, 0, 0] for _ in range(3)]
    assert "empty_confusion_matrix" in assess(data)["failures"]


def test_gate_rejects_negative_confusion_counts():
    data = report()
    data["transformer"]["confusion_matrix"][0][0] = -1
    assert "invalid_confusion_matrix" in assess(data)["failures"]
