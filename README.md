# Project 04 — Sentiment Intelligence

## v2.0 modernization — active, staged delivery

**Current release status: v2.0 in progress, not production approved.** The working baseline, CPU transformer integration, and 3-seed quality experiment remain intact. Version 2 work is incremental and CI-verified, not a claim that every enterprise capability below has been implemented.

### Verified experimental evidence

| Seed | TF-IDF macro-F1 | DistilBERT macro-F1 | Experimental gate |
|---|---:|---:|---|
| 42 | 0.4401 | 0.6464 | pass |
| 123 | 0.4425 | 0.6676 | pass |
| 2026 | 0.5271 | 0.6435 | pass |

These results use **1,200 train / 300 validation / 300 test** sampled TweetEval examples per seed, 3 epochs and CPU only; they are not full official benchmark results. [Run and artifacts](https://github.com/sundayayandele/ai-engineering-04-sentiment-intelligence/actions/runs/38395772963).

### New v2 controls

- `scripts/model_quality_gate.py` rejects missing/nonfinite metrics, class collapse, zero recall and regressions versus TF-IDF; run with `python scripts/model_quality_gate.py artifacts/quality-seed-42/comparison.json`.
- `scripts/model_release_manifest.py` generates SHA-256 hashes and file sizes for a **local** exported model; output remains explicitly staging-unapproved:

```bash
python scripts/model_release_manifest.py \
  --model-dir artifacts/tweet_eval_experiment/approved_model \
  --comparison artifacts/tweet_eval_experiment/comparison.json \
  --output artifacts/tweet_eval_experiment/staging-manifest.json
```

- Automated tests cover malformed quality reports and missing model weights. **The manifest is provenance metadata, not a cryptographic signature or registry approval.**

### v2 delivery roadmap

1. **Quality and reproducibility:** complete full official-split benchmark, validation-only tuning, dataset/model revision pinning, calibration and slice-level error analysis.
2. **Serving and security:** API versioning, structured audit-safe logs, OIDC/Keycloak, TLS and distributed quotas.
3. **Cloud-native operations:** tested Kubernetes/OpenShift manifests, Helm, Terraform/OpenTofu, image/SBOM scans and Prometheus/OpenTelemetry.
4. **Documentation and release:** C4/ArchiMate/TOGAF diagrams, runbooks, load/SLO evidence, signed model promotion and rollback rehearsals.

**Hardware and desktop labs:** Preserve the detailed Docker Desktop, Podman Desktop and Rancher Desktop setup guides below. Baseline learning host: 2–4 logical CPUs, 8 GiB minimum RAM and 15–25 GiB free disk. Transformer CPU experimentation: 4+ CPUs, 16 GiB preferred host RAM, 30+ GiB free disk; allocate ~8 GiB to a desktop VM/container engine when training inside it. No GPU is required for this phase.

---


**Project 04 of 20 Production AI Engineering Projects** · Python 3.12 · scikit-learn · FastAPI · optional Hugging Face/PyTorch · Docker/Podman/Kubernetes/OpenShift

> **Status:** Educational CPU baseline and enterprise reference scaffold. CI and the real-data baseline experiment have passed; bounded DistilBERT CPU verification passed; see measured results below. Synthetic training is not evidence of real sentiment accuracy; production deployment remains a reference.

## Two tracks

| Educational / Portfolio — runnable | Enterprise / Production — reference |
|---|---|
| Synthetic sentiment training, TF-IDF and Logistic Regression, FastAPI | Licensed data, transformer model comparison, MLOps, approved model registry |
| Docker Desktop, Rancher Desktop, Podman Desktop and Postman | Kubernetes/OpenShift, HA, IAM, TLS, observability and governance |
| GitHub Actions lint, tests, training, API smoke and container build | Promotion gates, private-cloud and cloud-neutral deployment patterns |

## Resource sizing

| Environment | CPU | RAM | Free disk |
|---|---|---|---|
| Single API container | 1 vCPU | 1 GiB | 5 GiB |
| Learner host | 2–4 logical CPUs | 8 GiB minimum, 16 GiB preferred | 15–25 GiB |
| Desktop engine VM (Windows/macOS) | 2–4 CPUs | 4 GiB minimum, 6–8 GiB preferred | 15–25 GiB |
| Enterprise reference | Benchmark driven | 2–4 GiB per replica starting estimate | 50+ GiB plus backups |

These are estimates, not validated production benchmarks. Baseline needs no GPU.

## Step 1 — Fork, clone and train

Fork this repository, then clone your fork and create a branch:

```bash
git clone https://github.com/YOUR_USERNAME/ai-engineering-04-sentiment-intelligence.git
cd ai-engineering-04-sentiment-intelligence
git switch -c learner/first-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/train_smoke.py
pytest -q
```

On Windows PowerShell use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`. Check `artifacts/sentiment_baseline.joblib` exists.

## Step 2A — Docker Desktop

Install Docker Desktop, start Linux containers, allocate 2+ CPUs / 4+ GiB memory / 15+ GiB disk where possible. Verify `docker version` and `docker compose version`:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=50
```

The committed Compose configuration binds localhost port 8000 and mounts the model read-only.

## Step 2B — Rancher Desktop

Install Rancher Desktop, select **dockerd (moby)**, start the engine, configure 2+ CPUs and 4+ GiB RAM, verify `docker version` and `docker compose version`. Use **the same Compose commands from Step 2A**.

## Step 2C — Podman Desktop

Install Podman Desktop, start the Podman machine on Windows/macOS, verify `podman version` and `podman info`. Run from repository root:

```bash
podman build -t project04-sentiment:local .
podman run -d --name project04-api -p 127.0.0.1:8000:8000 \
  --cpus 1 --memory 1g --security-opt no-new-privileges \
  -e MODEL_PATH=/app/artifacts/sentiment_baseline.joblib \
  -v "$PWD/artifacts:/app/artifacts:ro" project04-sentiment:local
podman ps
podman logs project04-api
```

## Step 3 — Postman API testing

1. GET `http://127.0.0.1:8000/health/live` and `/health/ready` — HTTP 200.
2. Open `http://127.0.0.1:8000/docs`.
3. Postman → **POST** `http://127.0.0.1:8000/predict` → **Body → raw → JSON**:

```json
{"text":"I love this product"}
```

4. Click Send. Expect `sentiment`, `confidence`, `probabilities` and `model_type`. This is a synthetic smoke model, not a reliable sentiment classifier.
5. Cleanup: `docker compose down` or `podman rm -f project04-api`.

## Step 4 — GitHub-hosted runners

1. Fork repo; create `learner/github-runner-lab`; review and enable Actions.
2. Push changes or choose **Actions → ci → Run workflow**.
3. Verify jobs `quality`, `education-smoke` and `container`. Record logs, SHA and run URL.
4. Standard public-repository runners may be free under GitHub's current policy; runners are ephemeral, not permanent hosting.

**Full step-by-step learner lab:** [Docker Desktop, Rancher Desktop, Podman Desktop, Postman and GitHub runners](docs/learning/DESKTOP-CONTAINERS-AND-RUNNERS-LAB.md).

## Real-data transformer milestone — TweetEval sentiment

A **reproducible experiment** now compares a classical TF-IDF + Logistic Regression baseline against fine-tuned Hugging Face DistilBERT on the **same official TweetEval sentiment test subset**. TweetEval sentiment is CC BY 3.0 according to the dataset card (other TweetEval tasks have different licenses). The experiment outputs accuracy, macro-F1, per-class metrics, confusion matrices, latency and CPU/GPU resource observations. A bounded real-data CPU experiment completed successfully; its small-sample metrics are documented below and are not production-quality evidence.

Install optional dependencies and run the baseline first:

```bash
python -m pip install -e ".[experiment]"
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --skip-transformer
```

Fine-tune DistilBERT on CPU (may be slow):

```bash
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --epochs 1 --batch-size 8 --device cpu
```

For CUDA-compatible NVIDIA hardware, install the appropriate PyTorch CUDA wheel and select `--device cuda`. Alternatively use **Actions → transformer-experiment → Run workflow** and select `baseline` or `transformer` (capped sample sizes). Results are stored in `artifacts/tweet_eval_experiment/comparison.json` locally or as a workflow artifact.

### Verified experiment results — GitHub-hosted CPU runner

**Run:** [transformer-experiment #38305611278](https://github.com/sundayayandele/ai-engineering-04-sentiment-intelligence/actions/runs/38305611278) (passed) · **commit:** `adca583b` · **source:** [original JSON artifact](https://github.com/sundayayandele/ai-engineering-04-sentiment-intelligence/actions/runs/38305611278/artifacts/11773086224).

Both models were evaluated on the **same 96-example TweetEval sentiment test subset** after 256 training examples (64 validation), seed 42. DistilBERT was fine-tuned for **one epoch** on CPU, batch size 4, max token length 96.

| Actual measured result | TF-IDF + Logistic Regression | DistilBERT (CPU) |
|---|---:|---:|
| Test accuracy | **42.71%** | 41.67% |
| Test macro-F1 | **0.3089** | 0.1961 |
| Training duration | **0.52 s** | 26.66 s |
| Mean inference latency (50 requests) | **0.78 ms** | 34.49 ms |
| p95 inference latency | **0.94 ms** | 43.87 ms |

The experiment process reached **2.12 GiB peak RSS**; this is not an isolated memory measurement for either model. The baseline was **44.2× faster** in mean local inference latency. DistilBERT predicted **neutral for every test example**, and the baseline failed to predict any negative examples. **Neither model is production-ready.** This experiment verifies the fine-tuning/evaluation workflow, not generalization or deployment performance.

### CPU versus GPU — measured versus estimated

| Deployment | Actual measurement | Cost implication |
|---|---|---|
| CPU TF-IDF | 0.78 ms mean inference; 0.52 s training | Lower compute demand for this tiny benchmark |
| CPU DistilBERT | 34.49 ms mean inference; 26.66 s training | Higher CPU cost per local inference if compute pricing is comparable |
| GPU DistilBERT | **Not tested** | No measured GPU latency, runtime, VRAM or cost |
| Cloud GPU vs CPU | **No vendor rate measured** | Price using region-specific VM hourly rates × measured runtime; include idle time and operations |

A transparent training-compute comparison is `cost = hourly_instance_rate × training_seconds / 3600`. GPU break-even requires a **measured GPU training duration**; publishing a numerical GPU saving now would be speculative. See [verified metrics, confusion matrices, hardware caveats and TCO methodology](docs/mlops/VERIFIED-CPU-RESULTS.md) and [CPU/GPU TCO framework](docs/infrastructure/TRANSFORMER-CPU-GPU-TCO.md).

**Detailed experiment, dataset license, methodology and reproducibility:** [Transformer experiment guide](docs/mlops/TRANSFORMER-EXPERIMENT.md).

## Next milestone — model quality and CPU transformer staging

**Implemented:** separate FastAPI transformer service, local-only `safetensors` loading, CPU Docker/Compose deployment, mock API tests and optional checkpoint export. **Not yet validated:** actual exported transformer checkpoint running end-to-end in a container; production model quality remains failed.

Run a larger **CPU-only** experiment with a saved staging artifact (runtime and memory requirements depend on your host):

```bash
python -m pip install -e ".[experiment]"
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --epochs 3 --batch-size 8 --device cpu --save-model
docker compose -f compose.transformer.yaml build
docker compose -f compose.transformer.yaml up -d
curl http://127.0.0.1:8001/health/ready
curl -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' -d '{"text":"I enjoyed this service"}'
```

The `approved_model` directory name is a **local mount convention only**: exporting a model does **not** confer production approval. The CPU staging container binds localhost port 8001; requires approximately 4 logical CPUs, 16 GiB RAM preferred and 30 GiB disk on the host. Docker Desktop and Rancher Desktop (moby) use the same Compose commands; Podman Desktop can build `Dockerfile.transformer` and run with a read-only artifact volume. No GPU purchase is required.

See [quality improvement plan, reproducibility and promotion gates](docs/mlops/QUALITY-AND-PROMOTION-GATES.md). **Do not promote the previous collapsed-neutral DistilBERT checkpoint.**

## Enterprise architecture and visual artifacts

- [TOGAF 10](docs/architecture/TOGAF10-ARCHITECTURE.md)
- [ArchiMate-inspired viewpoints](docs/architecture/ARCHIMATE-VIEWPOINTS.md)
- [C4 architecture](docs/architecture/C4.md)
- [Editable Mermaid architecture sources](diagrams/source/) (SVG and PNG renders committed)
- [Infrastructure sizing](docs/infrastructure/CAPACITY-PLANNING.md)
- [TCO comparison](docs/COST-TCO.md)
- [Dataset licensing](docs/datasets/DATASETS.md)
- [Security](docs/security/SECURITY.md)
- [Model card](docs/mlops/MODEL-CARD.md)
- [Kubernetes/OpenShift starter deployment](docs/DEPLOYMENT.md)

## API

- `GET /health/live`
- `GET /health/ready`
- `POST /predict` JSON text sentiment

## Scope

Enterprise diagrams include conceptual services not yet implemented. Before production use, validate real-world accuracy, licensing, security, latency, calibration, bias, HA and monitoring.

## Architectural Diagrams & Infographics
![Architectural Diagrams & Infographics](/diagrams/01-platform-overview.png)


![Architectural Diagrams & Infographics](/diagrams/02-enterprise-architecture.png)


![Architectural Diagrams & Infographics](/diagrams/03-nlp-pipeline-and-deployment.png)


## License

Project code intended for MIT; third-party dataset and pretrained model licenses remain separate.
