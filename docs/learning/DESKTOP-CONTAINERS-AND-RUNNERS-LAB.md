# Project 04 — Desktop labs and GitHub runners

## Hardware sizing (educational estimates)

| Workload | CPU | RAM | Free disk |
|---|---|---|---|
| API container | 1 vCPU | 1 GiB | 5 GiB |
| Learner host | 2–4 logical CPUs | 8 GiB minimum; 16 GiB preferred | 15–25 GiB |
| Docker/Rancher/Podman VM | 2–4 vCPU | 4 GiB minimum; 6–8 GiB preferred | 15–25 GiB |
| Transformer experimentation | 4+ CPU or GPU | 16+ GiB host | 30+ GiB |

These are planning values, not benchmarks. Baseline training uses no GPU.

## Step 1 — Fork, clone, train and test

Fork the repository to your account and replace YOUR_USERNAME:

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

Windows PowerShell: `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`. Confirm `artifacts/sentiment_baseline.joblib` exists. Synthetic training does not establish real-world accuracy.

## Step 2A — Docker Desktop

Install Docker Desktop, start the Linux container engine, and allocate at least 2 CPUs, 4 GiB memory and 15 GiB disk where supported. With WSL2, VM limits may be controlled by WSL settings. Verify `docker version` and `docker compose version`. In the repository root:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=50
```

The Compose file mounts `artifacts/` read-only, uses localhost port 8000 and limits the API container to 1 CPU and 1 GiB memory.

## Step 2B — Rancher Desktop

Install Rancher Desktop, select the **dockerd (moby)** engine, start it, configure at least 2 CPUs and 4 GiB memory, and verify `docker version` and `docker compose version`. Run the same Compose commands from Step 2A. Optionally enable Kubernetes for cluster learning; a single desktop is not production HA.

## Step 2C — Podman Desktop

Install Podman Desktop and complete onboarding. Start the Podman machine on Windows/macOS; on Linux use the rootless Podman engine. Verify `podman version` and `podman info`. In the repository root:

```bash
podman build -t project04-sentiment:local .
podman run -d --name project04-api \
  -p 127.0.0.1:8000:8000 --cpus 1 --memory 1g \
  --security-opt no-new-privileges \
  -e MODEL_PATH=/app/artifacts/sentiment_baseline.joblib \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  project04-sentiment:local
podman ps
podman logs project04-api
podman stats --no-stream project04-api
```

On SELinux-enforcing Linux, the volume may require `:ro,Z`. On Windows/macOS, ensure the host path is shared with the Podman machine and adapt PowerShell quoting as needed.

## Step 3 — Postman

1. Open `http://127.0.0.1:8000/health/live` (HTTP 200).
2. Open `http://127.0.0.1:8000/health/ready` (HTTP 200 after model loads).
3. Open `http://127.0.0.1:8000/docs`.
4. In Postman: **POST** `http://127.0.0.1:8000/predict` → **Body → raw → JSON**:

```json
{"text":"I love this product"}
```

5. Click Send. Expect `sentiment`, `confidence`, `probabilities` and `model_type`. Test an empty string to observe validation error HTTP 422.

Equivalent curl:

```bash
curl -i -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"I love this product"}'
```

## Step 4 — GitHub-hosted runners

1. Fork and create a `learner/github-runner-lab` branch.
2. Review `.github/workflows/ci.yml`; enable Actions in your fork.
3. Push changes or select **Actions → ci → Run workflow**.
4. Inspect `quality`, `education-smoke` and `container` jobs. The smoke job trains the model and tests a live FastAPI prediction through TestClient.
5. Save logs, commit SHA, job URL and screenshots as portfolio evidence. Standard public-repository runners may be free subject to GitHub policy; runners are ephemeral and not hosting.

## Step 5 — Cleanup and troubleshooting

Docker/Rancher: `docker stats --no-stream`, `docker compose logs --tail=100`, `docker compose down`.

Podman: `podman logs project04-api`, `podman rm -f project04-api`.

| Symptom | Check |
|---|---|
| Readiness 503 | Model artifact generated and mounted; inspect logs |
| Port 8000 in use | Stop other server or change host port |
| Postman HTTP 422 | JSON body must contain nonempty `text` |
| Podman volume denied | Machine file sharing, rootless UID, SELinux relabel |
| Container out of memory | Increase desktop VM allocation |
| Unreliable sentiment | Synthetic lab data is not real-world evaluation |

## Enterprise boundary

Before production: licensed data, independent test sets, transformer comparison, fairness/calibration, authenticated TLS gateway, artifact signing, monitoring, multiple failure domains and workload-based sizing. See [capacity planning](../infrastructure/CAPACITY-PLANNING.md) and [TCO](../COST-TCO.md).
