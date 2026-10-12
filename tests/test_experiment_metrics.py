from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location("experiment", Path(__file__).resolve().parents[1] /
                               "scripts" / "compare_transformer.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_metrics_three_class():
    report = module.metric_report([0, 1, 2, 0], [0, 1, 1, 0])
    assert report["accuracy"] == 0.75
    assert 0 <= report["macro_f1"] <= 1
    assert len(report["confusion_matrix"]) == 3


def test_latency_report():
    result = module.benchmark(lambda x: [0] * len(x), ["test"] * 5)
    assert result["requests"] == 5
    assert result["p95_ms"] >= 0
