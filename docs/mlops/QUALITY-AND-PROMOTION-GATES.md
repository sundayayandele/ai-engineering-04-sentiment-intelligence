# Model quality improvement and promotion gates

**Status:** transformer training and local CPU serving path are implemented for staging experiments, but no trained checkpoint is approved for production.

## Quality improvement

1. Run a **larger, class-stratified** TweetEval train/validation/test experiment. Keep the official test partition untouched for final evaluation; tune using validation only.
2. Use 3 epochs initially, compare learning rates and sequence lengths, log dataset/model revisions and random seeds. Run at least 3 seeds.
3. Record per-class precision, recall and F1, macro-F1, confusion matrix, latency, memory, and calibration. Reject single-class collapse.
4. Benchmark against a class-weighted TF-IDF baseline and a majority-class dummy classifier. Choose model only if business-weighted metrics and latency justify complexity.
5. Save a reviewed safetensors checkpoint and tokenizer locally; do not commit large model weights or confidential data.
6. Validate on realistic, separately curated customer feedback and multilingual/dialect slices as appropriate.

## Promotion policy

Staging requires a saved checkpoint, reproducible metrics, all three classes predicted, passing API integration tests and a signed/approved model artifact. Production requires **business-approved quantitative thresholds**, data and model license review, security scan, auth/TLS/rate limits, PII controls, measured p95/p99 under target concurrency, staging soak test, rollback and documented approval. None is currently established.

## Local CPU transformer serving

After an experiment saves an artifact in `artifacts/tweet_eval_experiment/approved_model`:

```bash
export TRANSFORMER_API_KEY=replace-with-random-secret-at-least-24-chars
docker compose -f compose.transformer.yaml build
docker compose -f compose.transformer.yaml up -d
curl http://127.0.0.1:8001/health/ready
curl -X POST http://127.0.0.1:8001/predict -H "X-API-Key: $TRANSFORMER_API_KEY" -H 'Content-Type: application/json' -d '{"text":"I enjoyed this service"}'
```

On Podman Desktop use `podman build -f Dockerfile.transformer -t project04-transformer:cpu .` and `podman run` with localhost 8001, a read-only approved-model mount, 2 CPU and 4 GiB memory. On Rancher Desktop with moby, use the Docker Compose commands. Host sizing: 4+ logical CPUs, 16 GiB RAM preferred, 30+ GiB free disk. **No GPU required.**

This service loads only local model files with safetensors and remote code disabled. It requires a strong API key, uses an in-memory per-client rate limiter and binds to localhost. For production, implement TLS termination, central authentication, distributed rate limiting and security validation.

## Reproducible three-seed CPU quality study

GitHub Actions workflow `.github/workflows/cpu-model-quality-experiment.yml` runs seeds **42, 123 and 2026** independently on hosted CPU runners, with 1,200 class-stratified training samples, 300 validation and 300 test examples, 3 epochs, batch size 8 and max sequence length 96. Each seed preserves TweetEval's official split boundaries and publishes `comparison.json` and `quality-gate.json` as workflow artifacts for 30 days. These are exploratory samples, **not** the full TweetEval benchmark.

For a local repeat, with Python 3.12 and at least 4 CPU cores, 16 GiB RAM preferred, and 30 GiB free disk:

```bash
python -m pip install -e ".[experiment]"
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --epochs 3 --batch-size 8 --max-length 96 --seed 42 --device cpu --stratified --output-dir artifacts/quality-seed-42
python scripts/model_quality_gate.py artifacts/quality-seed-42/comparison.json --output artifacts/quality-seed-42/quality-gate.json
```

The experimental gate requires macro-F1 >= 0.40, no zero-recall sentiment class, predictions for all three classes, and transformer macro-F1 at least the TF-IDF baseline's. A failing gate is reported without suppressing the experiment artifacts or declaring a production-ready model. Thresholds are initial *engineering screening* values, not customer-approved production criteria. Repeat on the full official test set only after selecting hyperparameters from validation results.

For **Docker Desktop**, **Podman Desktop**, and **Rancher Desktop**, the preferred experiment environment is a local Python virtual environment (training in a desktop container is optional). Allocate at least 4 logical CPUs, 16 GiB host RAM preferred, 8 GiB container/VM memory when running training inside a container, and 30 GiB free disk; increase memory if the dataset/model demands it. For inference use the existing read-only model mount and platform-specific instructions above. No GPU purchase is required.
