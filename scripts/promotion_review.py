"""Fail-closed review decision for model promotion; never deploys a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_promotion(gate: dict, manifest: dict, approval: dict) -> dict:
    reasons = []
    if gate.get("status") != "pass":
        reasons.append("quality_gate_not_passed")
    if gate.get("production_approved") is not False:
        reasons.append("gate_provenance_invalid")
    if manifest.get("stage") != "staging-unapproved" or manifest.get("production_approved") is not False:
        reasons.append("manifest_provenance_invalid")
    if not manifest.get("files") or not manifest.get("seed"):
        reasons.append("manifest_incomplete")
    if approval.get("decision") != "approve":
        reasons.append("explicit_approval_missing")
    if not isinstance(approval.get("reviewer"), str) or not approval["reviewer"].strip():
        reasons.append("reviewer_missing")
    if not isinstance(approval.get("ticket"), str) or not approval["ticket"].strip():
        reasons.append("approval_ticket_missing")
    if approval.get("artifact_sha256") != manifest.get("files", {}).get("model.safetensors", {}).get("sha256"):
        reasons.append("approved_artifact_digest_mismatch")
    return {
        "status": "eligible-for-controlled-staging-promotion" if not reasons else "blocked",
        "production_approved": False,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_promotion(*(json.loads(path.read_text()) for path in
                                  (args.gate, args.manifest, args.approval)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not result["reasons"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
