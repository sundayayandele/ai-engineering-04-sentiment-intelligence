"""Release manifest checks avoid heavyweight model dependencies."""
import json

import pytest

from scripts.model_release_manifest import manifest


def test_manifest_hashes_artifacts_without_approving_production(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model / name).write_bytes(b"example")
    metrics = tmp_path / "comparison.json"
    metrics.write_text(json.dumps({
        "seed": 42, "dataset": "tweet_eval",
        "transformer": {"macro_f1": 0.65},
    }))
    result = manifest(model, metrics)
    assert result["production_approved"] is False
    assert result["stage"] == "staging-unapproved"
    assert len(result["files"]["model.safetensors"]["sha256"]) == 64


def test_manifest_rejects_missing_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="model.safetensors"):
        manifest(model, tmp_path / "comparison.json")
