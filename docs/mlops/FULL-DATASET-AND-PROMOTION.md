# Full-dataset benchmark and controlled promotion (v2.0)

**Current state:** implemented tooling, not a completed full-dataset training result. Project 05 remains on hold. Experimental three-seed sampled results do **not** qualify a model for production.

## Benchmark execution

The opt-in GitHub Actions workflow `full-dataset-cpu-benchmark.yml` requires a dedicated Linux x64 runner with labels `self-hosted,linux,x64,full-dataset-cpu`. Register that label **only** on an approved machine. The workflow is deliberately manual because full DistilBERT CPU fine-tuning can take hours; do not assume a GitHub-hosted runner will complete. Suggested starting capacity: 8+ vCPUs, 32 GiB RAM, 60+ GiB free SSD. Measure actual usage.

Select Actions → **full-dataset-cpu-benchmark** → Run workflow, choose seed and epochs. Repeat seeds 42, 123 and 2026. The script's `--train-samples 0 --validation-samples 0 --test-samples 0` means **use all examples in each official split**. It never mixes the splits.

Local execution (Python 3.12):

```bash
python -m pip install -e ".[experiment]"
python scripts/compare_transformer.py --train-samples 0 --validation-samples 0 --test-samples 0 --epochs 2 --batch-size 8 --max-length 96 --seed 42 --device cpu --output-dir artifacts/full-seed-42
python scripts/model_quality_gate.py artifacts/full-seed-42/comparison.json --output artifacts/full-seed-42/quality-gate.json
```

Repeat with the other seeds. The output includes uncalibrated validation-set ECE, Brier score, negative log-likelihood and confidence bins. **Calibration diagnostics are not temperature scaling**; a fitted calibrator and held-out evaluation remain future work. Choose hyperparameters using validation only; report the official test results after model selection, not for iterative tuning.

Docker Desktop, Podman Desktop and Rancher Desktop remain supported for serving and learner labs per README. For CPU training in a desktop VM, allocate 8+ vCPUs and 16–24 GiB VM memory where available, and 60+ GiB disk; a native Python environment or dedicated runner is simpler for large training.

## Controlled model promotion

1. Verify quality metrics and complete per-class, fairness/slice, licensing, PII and security reviews.
2. Export an **approved-for-review, not approved-for-production** safetensors checkpoint with `--save-model` (prefer a separate reproducible training/export run).
3. Generate the SHA-256 staging manifest using `scripts/model_release_manifest.py`.
4. Create a **private**, reviewed approval JSON with `decision: approve`, nonempty `reviewer`, `ticket` and `artifact_sha256` matching the manifest's `model.safetensors` digest.
5. Run `python scripts/promotion_review.py --gate quality-gate.json --manifest staging-manifest.json --approval approval.json --output promotion-decision.json`.

This review tool yields **staging eligibility only**, not deployment or production approval. Do not commit reviewer approval records, model weights or datasets. Production requires signed immutable artifacts, separate authorized approvers, environment protection, provenance attestation, security gates, observability, load/soak evidence and rollback verification. The current JSON approval check is not a substitute for identity-backed authorization.

## v2.0 acceptance criteria

- Full official-split results for at least three reproducible seeds, with confidence intervals and class-level diagnostics
- Validation-only calibration/tuning and final test-set evaluation
- Model registry lineage and independently approved signed promotion
- Tested secure API and deployment on Docker/Podman, Kubernetes and OpenShift
- Observability, load/security testing, runbooks, C4/ArchiMate/TOGAF diagrams, resource sizing and rollback drill

**Do not start Project 05 until these criteria are evidenced and reviewed.**
