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
