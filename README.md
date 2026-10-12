# Project 04 — Sentiment Intelligence

**Project 04 of 20 Production AI Engineering Projects** · Python 3.12 · scikit-learn · FastAPI · optional Hugging Face/PyTorch · Docker/Podman/Kubernetes/OpenShift

> **Status:** Educational CPU baseline and enterprise reference scaffold. CI and the real-data baseline experiment have passed; bounded DistilBERT verification is underway. Synthetic training is not evidence of real sentiment accuracy; production deployment remains a reference.

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

A **reproducible experiment** now compares a classical TF-IDF + Logistic Regression baseline against fine-tuned Hugging Face DistilBERT on the **same official TweetEval sentiment test subset**. TweetEval sentiment is CC BY 3.0 according to the dataset card (other TweetEval tasks have different licenses). The experiment outputs accuracy, macro-F1, per-class metrics, confusion matrices, latency and CPU/GPU resource observations. No real-data benchmark results are claimed until a run completes.

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

**Detailed experiment, dataset license, methodology and reproducibility:** [Transformer experiment guide](docs/mlops/TRANSFORMER-EXPERIMENT.md).

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
