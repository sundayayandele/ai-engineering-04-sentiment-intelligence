from scripts.promotion_review import evaluate_promotion


def fixtures():
    digest = "a" * 64
    return (
        {"status": "pass", "production_approved": False},
        {"stage": "staging-unapproved", "production_approved": False,
         "seed": 42, "files": {"model.safetensors": {"sha256": digest}}},
        {"decision": "approve", "reviewer": "reviewer@example.org",
         "ticket": "CHANGE-123", "artifact_sha256": digest},
    )


def test_explicit_review_yields_staging_eligibility_only():
    result = evaluate_promotion(*fixtures())
    assert result["status"] == "eligible-for-controlled-staging-promotion"
    assert result["production_approved"] is False


def test_no_approval_fails_closed():
    gate, manifest, approval = fixtures()
    approval.pop("decision")
    assert "explicit_approval_missing" in evaluate_promotion(gate, manifest, approval)["reasons"]


def test_digest_mismatch_fails_closed():
    gate, manifest, approval = fixtures()
    approval["artifact_sha256"] = "b" * 64
    assert "approved_artifact_digest_mismatch" in evaluate_promotion(
        gate, manifest, approval
    )["reasons"]
